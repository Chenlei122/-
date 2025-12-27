# -*- coding: utf-8 -*-
import argparse
import datetime
import re
import sys
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, RetryError, ServiceUnavailable

# ---------------------------------------------------------
# 1. 嘗試載入外部 AI
# ---------------------------------------------------------
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except ImportError:
    print("⚠️ 警告: 找不到 predict_and_advise.py，將使用內建簡易規則。")

# ---------------------------------------------------------
# 2. 輔助工具
# ---------------------------------------------------------
_NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")

def to_float(x, d: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        m = _NUMERIC_LIKE.search(str(x))
        return float(m.group()) if m else d
    except Exception: return d

def rule_based_fallback(data: Dict[str, Any]) -> Dict[str, Any]:
    steps = to_float(data.get("steps"), 0)
    tips = ["數據顯示活動量偏低，起來走走吧 🚶"] if steps < 2000 else ["一切正常！"]
    return {
        "advice": {"state": "ok", "tips": tips, "model_version": "fallback"},
        "_advised": True
    }

def generate_payload(data: Dict[str, Any], model_path: str, user_id: str) -> Dict[str, Any]:
    if _HAVE_MAKE_ADVICE:
        try:
            result = _external_make_advice(data, model_path=model_path)
        except Exception:
            result = rule_based_fallback(data)
    else:
        result = rule_based_fallback(data)
    
    result["user_id"] = user_id
    result["created_at"] = firestore.SERVER_TIMESTAMP
    return result

# ---------------------------------------------------------
# 3. Firestore 核心
# ---------------------------------------------------------
def init_firestore(key_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def batch_write_safe(batch, commit=True):
    if commit:
        try:
            batch.commit()
            return True
        except Exception as e:
            print(f"❌ 批次寫入失敗: {e}")
            return False
    return False

def process_user_date(db, user_id, date_str, model_path, args) -> tuple:
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    target_ref = db.collection("User_advice_results")

    # === V8 修改：加入 timeout 防止卡死 ===
    try:
        if args.verbose:
            print(f"    👉 正在連線 Firestore 讀取 {date_str}...", end="", flush=True)
        
        # 設定 5 秒超時，防止無限重試
        docs = list(source_ref.stream(timeout=5))
        
        if args.verbose:
            print(" ✅ 連線成功")
            
    except (ResourceExhausted, RetryError):
        print("\n❌ 讀取失敗: 額度已耗盡 (Quota Exceeded) 或連線被拒絕。請明天再試。")
        # 遇到這種錯誤，直接中斷程式，因為再試也沒用
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 讀取發生未知錯誤: {e}")
        return 0, 0, 0

    if not docs:
        return 0, 0, 0

    batch = db.batch()
    count = 0
    skipped = 0
    processed = 0

    for doc in docs:
        data = doc.to_dict()
        if (not args.force) and (data.get("_advised") is True):
            skipped += 1
            continue

        payload = generate_payload(data, model_path, user_id)
        if not payload: continue

        new_doc_id = f"{user_id}_{date_str}_{doc.id}"
        new_doc_ref = target_ref.document(new_doc_id)

        if not args.dry_run:
            batch.set(new_doc_ref, payload, merge=True)
            batch.update(doc.reference, {"_advised": True})
        
        processed += 1
        count += 1

        if count >= 400 and not args.dry_run:
            batch_write_safe(batch)
            batch = db.batch()
            count = 0

    if count > 0 and not args.dry_run:
        batch_write_safe(batch)

    return len(docs), skipped, processed

# ---------------------------------------------------------
# 4. 主程式 (V8: 防卡死 Debug 版)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--days", type=int, default=3, help="往回檢查的天數 (預設 3 天)")
    
    default_user = "j3tgphl0e9gj2hPRaT6aAOhH1922"
    parser.add_argument("--target-user", type=str, default=default_user)
    
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # 預設開啟 verbose 以便除錯
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (V8: 防卡死偵錯版)")
    print("="*60)
    print(f"👤 目標使用者: {args.target_user}")
    print(f"📅 檢查範圍:   過去 {args.days} 天")
    print(f"📂 輸出位置:   User_advice_results")
    print("-" * 60)

    total_processed_global = 0
    checked_count = 0

    today = datetime.date.today()
    target_dates = []
    for i in range(args.days + 1): 
        d = today - datetime.timedelta(days=i)
        target_dates.append(d.strftime("%Y-%m-%d"))

    for date_str in target_dates:
        print(f"🔍 [檢查] 日期: {date_str}")
            
        total, skipped, processed = process_user_date(
            db, args.target_user, date_str, args.model, args
        )
        
        if total > 0:
            print(f"   🎉 發現 {total} 筆紀錄 -> 新增 {processed} 筆 (略過 {skipped})")
            total_processed_global += processed
            checked_count += 1
        else:
            print(f"   💤 無資料")

    print("-" * 60)
    print(f"✅ 全部完成！")
    print(f"   - 總共新增建議: {total_processed_global}")

if __name__ == "__main__":
    main()
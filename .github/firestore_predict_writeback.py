# -*- coding: utf-8 -*-
import argparse
import datetime
import re
import sys
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import firebase_admin
from firebase_admin import credentials, firestore

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
    # 直接指定路徑，不管父資料夾是否為幽靈，只要 records 存在就能讀
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    target_ref = db.collection("User_advice_results")

    try:
        # 這裡只會讀取該使用者、該日期的資料，非常節省額度
        docs = list(source_ref.stream())
    except Exception:
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
# 4. 主程式 (V7: 狙擊手模式 / 省額度版)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    # 這裡的 days 變成「要往回檢查幾天」
    parser.add_argument("--days", type=int, default=30, help="往回檢查的天數 (預設 30 天)")
    
    default_user = "j3tgphl0e9gj2hPRaT6aAOhH1922"
    parser.add_argument("--target-user", type=str, default=default_user)
    
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (V7: 省流量狙擊版)")
    print("="*60)
    print(f"👤 目標使用者: {args.target_user}")
    print(f"📅 檢查範圍:   過去 {args.days} 天 (自動生成日期路徑)")
    print(f"📂 輸出位置:   User_advice_results")
    print("-" * 60)

    total_processed_global = 0
    checked_count = 0

    # === V7 核心邏輯：直接生成日期字串，不掃描全庫 ===
    today = datetime.date.today()
    
    # 產生從今天開始往回推 N 天的日期列表
    target_dates = []
    for i in range(args.days + 1): # +1 包含今天
        d = today - datetime.timedelta(days=i)
        target_dates.append(d.strftime("%Y-%m-%d"))

    # 針對每一個日期，直接去敲門
    for date_str in target_dates:
        if args.verbose:
            print(f"🔍 檢查 {date_str}...", end="\r")
            
        total, skipped, processed = process_user_date(
            db, args.target_user, date_str, args.model, args
        )
        
        if total > 0:
            print(f"📅 {date_str}: 發現 {total} 筆紀錄 -> ✅ 新增 {processed} 筆 (略過 {skipped})")
            total_processed_global += processed
            checked_count += 1
        else:
            # 沒資料也是正常的，不報錯，安靜略過
            pass

    print("-" * 60)
    print(f"✅ 全部完成！")
    print(f"   - 檢查了最近 {args.days} 天")
    print(f"   - 實際有資料的天數: {checked_count}")
    print(f"   - 總共新增建議: {total_processed_global}")
    print(f"   - 狀態: 額度安全 🛡️")

if __name__ == "__main__":
    main()
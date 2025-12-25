# -*- coding: utf-8 -*-
import argparse
import datetime
import time
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

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
    # 簡易規則備案
    steps = to_float(data.get("steps"), 0)
    tips = ["數據顯示活動量偏低，起來走走吧 🚶"] if steps < 2000 else ["一切正常！"]
    return {
        "advice": {"state": "ok", "tips": tips, "model_version": "fallback"},
        "_advised": True
    }

def generate_payload(data: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    if _HAVE_MAKE_ADVICE:
        try:
            return _external_make_advice(data, model_path=model_path)
        except Exception:
            pass
    return rule_based_fallback(data)

# ---------------------------------------------------------
# 3. Firestore 核心
# ---------------------------------------------------------
def init_firestore(key_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def get_date_range(days):
    base = datetime.date.today()
    # 往前多抓幾天，確保能涵蓋你的測試資料
    return [(base - datetime.timedelta(days=i)).isoformat() for i in range(-2, days)]

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
    # 建構路徑: users/{uid}/health_data/{date}/records
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    target_ref = db.collection("users").document(user_id).collection("advice_results")

    try:
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

        payload = generate_payload(data, model_path)
        if not payload: continue

        new_doc_id = f"{date_str}_{doc.id}"
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
# 4. 主程式 (DEBUG 版本)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 4. 主程式 (修正版：雙層迴圈精確掃描)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--days", type=int, default=7) # 這裡保留參數但不使用，改為自動掃描所有日期
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (精確掃描版)")
    print(f"🔗 連接專案 ID: {db.project}")
    print("="*60)

    total_processed_global = 0
    
    # 1. 第一層：掃描 users 集合
    print("🔍 正在讀取所有使用者清單...")
    try:
        users_ref = db.collection("users").stream()
    except Exception as e:
        print(f"❌ 讀取 users 失敗: {e}")
        return

    found_any_user = False

    for user_doc in users_ref:
        user_id = user_doc.id
        found_any_user = True
        print(f"\n👤 正在檢查使用者: {user_id}")

        # 2. 第二層：直接掃描該使用者的 health_data 裡有哪些日期
        # 路徑: users/{user_id}/health_data
        health_ref = db.collection("users").document(user_id).collection("health_data").stream()
        
        found_dates = False
        for date_doc in health_ref:
            found_dates = True
            date_str = date_doc.id
            
            # 這裡呼叫原本寫好的處理函式
            # 它會去抓 records 並生成建議
            total, skipped, processed = process_user_date(db, user_id, date_str, args.model, args)
            
            if total > 0:
                print(f"   📅 日期 {date_str}: 原始資料 {total} 筆 -> 新增建議 {processed} 筆 (略過 {skipped})")
                total_processed_global += processed
            elif args.verbose:
                print(f"   📅 日期 {date_str}: 無有效 records 資料")

        if not found_dates:
            print("   ⚠️  (此使用者底下沒有 health_data 資料夾)")

    if not found_any_user:
        print("❌ 找不到任何使用者 (users 集合是空的，或是權限不足)")

    print("-" * 60)
    print(f"✅ 任務完成！總計新增建議: {total_processed_global} 筆")

if __name__ == "__main__":
    main()
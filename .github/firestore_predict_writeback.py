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
    # 產生建議
    if _HAVE_MAKE_ADVICE:
        try:
            result = _external_make_advice(data, model_path=model_path)
        except Exception:
            result = rule_based_fallback(data)
    else:
        result = rule_based_fallback(data)
    
    # 在資料中標記是屬於哪個使用者的
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
    # 來源路徑: users/{uid}/health_data/{date}/records
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    # === 🎯 修改點：集合名稱改成 User_advice_results ===
    target_ref = db.collection("User_advice_results")

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

        # 產生內容 (帶入 user_id)
        payload = generate_payload(data, model_path, user_id)
        if not payload: continue

        # 檔名格式: UserID_日期_時間 (確保唯一性)
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
# 4. 主程式 (V6: 指定集合名稱版)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--days", type=int, default=7, help="相容性參數(不使用)")
    
    default_user = "j3tgphl0e9gj2hPRaT6aAOhH1922"
    parser.add_argument("--target-user", type=str, default=default_user)
    
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (V6: User_advice_results)")
    print("="*60)
    print(f"👤 目標使用者: {args.target_user}")
    print(f"📂 輸出位置:   最外層集合 `User_advice_results`")
    print("-" * 60)

    # 1. 幽靈獵人模式：全域掃描 records
    print(f"🔍 正在深度掃描 records...")
    try:
        all_records = db.collection_group("records").stream()
    except Exception as e:
        print(f"❌ 搜尋 records 失敗: {e}")
        return

    found_dates = set()
    scanned_count = 0

    # 2. 找出該使用者的所有日期
    for doc in all_records:
        path = doc.reference.path
        parts = path.split("/")
        
        if args.target_user in parts:
            scanned_count += 1
            try:
                if "health_data" in parts:
                    date_idx = parts.index("health_data") + 1
                    date_str = parts[date_idx]
                    found_dates.add(date_str)
            except Exception:
                pass

    if not found_dates:
        print("❌ 找不到資料，請確認 ID 是否正確。")
        return

    sorted_dates = sorted(list(found_dates))
    print(f"📊 找到 {len(sorted_dates)} 個日期，開始生成建議...")

    # 3. 處理
    total_processed_global = 0
    for date_str in sorted_dates:
        total, skipped, processed = process_user_date(
            db, args.target_user, date_str, args.model, args
        )
        if total > 0:
            print(f"📅 {date_str}: {total} 筆資料 -> ✅ 已寫入 Root/User_advice_results ({processed} 筆)")
            total_processed_global += processed

    print("-" * 60)
    print(f"✅ 完成！共 {total_processed_global} 筆建議已儲存至最外層 `User_advice_results` 集合。")

if __name__ == "__main__":
    main()
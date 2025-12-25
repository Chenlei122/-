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
# 4. 主程式 (V3: 葉節點反查版)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--days", type=int, default=7) 
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (V3: 葉節點反查版)")
    print(f"🔗 連接專案 ID: {db.project}")
    print("="*60)

    # 1. 搜尋所有 records (這是我們唯一確定能抓到的東西)
    print("🔍 正在全域搜尋 'records' 集合 (這可能需要一點時間)...")
    try:
        # 先抓 500 筆作為樣本，避免資料量過大卡住 (若確認無誤可移除 limit)
        record_docs = db.collection_group("records").limit(500).stream()
    except Exception as e:
        print(f"❌ 搜尋 records 失敗: {e}")
        return

    tasks = set() # 用來儲存 (User, Date) 的唯一組合
    scanned_count = 0

    print("📊 正在解析路徑結構...")
    for doc in record_docs:
        scanned_count += 1
        full_path = doc.reference.path
        # 路徑範例應該是: users/{UID}/health_data/{DATE}/records/{ID}
        
        parts = full_path.split("/")
        
        # 顯示第一筆資料的路徑，用來除錯
        if scanned_count == 1:
            print(f"   [路徑樣本] {full_path}")
            print(f"   [路徑切分] {parts}")

        # 邏輯: 尋找 'users' 和 'health_data' 的相對位置
        try:
            # 確保結構中有 users
            if "users" in parts and "health_data" in parts:
                u_idx = parts.index("users") + 1
                h_idx = parts.index("health_data")
                
                # User ID 應該在 users 後面
                user_id = parts[u_idx]
                
                # Date 應該在 health_data 後面
                if h_idx + 1 < len(parts):
                    date_str = parts[h_idx + 1]
                    tasks.add((user_id, date_str))
        except Exception:
            continue

    print(f"🔎 掃描了 {scanned_count} 筆紀錄，整理出 {len(tasks)} 個待處理任務")
    
    if len(tasks) == 0 and scanned_count > 0:
        print("❌ 路徑解析失敗！請截圖上面的 [路徑樣本] 給我看。")
        return

    print("-" * 60)

    # 2. 執行任務
    sorted_tasks = sorted(list(tasks), key=lambda x: x[0])
    total_processed_global = 0
    current_user = None

    for user_id, date_str in sorted_tasks:
        if user_id != current_user:
            print(f"\n👤 正在處理使用者: {user_id}")
            current_user = user_id
            
        total, skipped, processed = process_user_date(db, user_id, date_str, args.model, args)
        
        if total > 0:
            print(f"   📅 日期 {date_str}: 資料 {total} 筆 -> 新增 {processed} 筆 (略過 {skipped})")
            total_processed_global += processed

    print("-" * 60)
    print(f"✅ 任務完成！總計新增建議: {total_processed_global} 筆")

if __name__ == "__main__":
    main()
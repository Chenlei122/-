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
    dates = get_date_range(args.days)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (路徑顯影 DEBUG 版)")
    print(f"🔗 連接專案 ID: {db.project}")
    print("="*60)

    user_ids = []
    
    # 直接使用最強力的 Collection Group Query
    print("🔍 啟動深度掃描 (records)...")
    try:
        blobs = list(db.collection_group("records").limit(20).stream())
        print(f"   📊 掃描到 {len(blobs)} 筆原始資料")
        
        found_set = set()
        for i, b in enumerate(blobs):
            path = b.reference.path
            # 印出前 3 筆資料的完整路徑，讓我們看看長什麼樣子
            if i < 3:
                print(f"   [DEBUG路徑] {path}")

            # 解析邏輯：嘗試多種方法抓取 User ID
            parts = path.split("/")
            
            # 方法 1: 標準結構 users/{UID}/...
            if "users" in parts:
                idx = parts.index("users")
                if idx + 1 < len(parts):
                    found_set.add(parts[idx + 1])
                    continue
            
            # 方法 2: 硬解 (假設它是第 2 層，索引為 1)
            # users(0) / UID(1) / health_data(2) / ...
            if len(parts) >= 2:
                 # 簡單檢查一下這個 ID 長得像不像 User ID (不是 users 或 health_data)
                 candidate = parts[1]
                 if candidate not in ["users", "health_data", "records"]:
                     found_set.add(candidate)

        user_ids = list(found_set)

    except Exception as e:
        print(f"❌ 掃描失敗: {e}")

    if not user_ids:
        print("❌ 依然找不到 User ID。請截圖上面的 [DEBUG路徑] 給我看。")
        return

    print(f"👥 成功鎖定 {len(user_ids)} 位使用者: {user_ids}")
    
    total_processed_global = 0
    for uid in user_ids:
        print(f"\n👤 正在處理: {uid}")
        for d in tqdm(dates, desc=f"   Days"):
            _, _, proc = process_user_date(db, uid, d, args.model, args)
            total_processed_global += proc
            
    print("-" * 60)
    print(f"✅ 任務完成！新增建議: {total_processed_global} 筆")

if __name__ == "__main__":
    main()
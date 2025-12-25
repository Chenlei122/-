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
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (穿透式掃描版)")
    print(f"🔗 連接專案 ID: {db.project}")
    print("="*60)

    total_processed_global = 0
    
    # 使用 collection_group 直接搜尋所有的 "health_data" 集合中的文件
    # 這會回傳像是 users/{UID}/health_data/{DATE} 這樣的文件
    print("🔍 正在穿透搜尋全域 health_data 資料...")
    
    try:
        # 這裡抓到的是日期文件 (例如 2025-04-07)
        date_docs = db.collection_group("health_data").stream()
    except Exception as e:
        print(f"❌ 搜尋失敗: {e}")
        return

    # 為了避免重複處理，我們先收集所有任務
    # 格式: (user_id, date_str)
    tasks = set()

    found_count = 0
    for doc in date_docs:
        found_count += 1
        date_str = doc.id
        full_path = doc.reference.path # 例如: users/abc12345/health_data/2025-04-07
        
        # 解析路徑以取得 User ID
        path_parts = full_path.split("/")
        
        # 確保路徑結構正確：必須包含 users
        if "users" in path_parts:
            # 找到 "users" 的下一個元素就是 User ID
            uid_idx = path_parts.index("users") + 1
            if uid_idx < len(path_parts):
                user_id = path_parts[uid_idx]
                tasks.add((user_id, date_str))
        else:
            # 如果資料不在 users 底下 (例如在根目錄)，視情況處理或略過
            if args.verbose:
                print(f"⚠️ 發現非標準路徑資料 (略過): {full_path}")

    print(f"📊 掃描完成，共發現 {found_count} 個日期資料夾")
    print(f"👥 整理出 {len(tasks)} 個待處理任務")
    print("-" * 60)

    # 開始執行處理
    # 依使用者分組顯示，比較好看
    sorted_tasks = sorted(list(tasks), key=lambda x: x[0])
    
    current_user = None
    for user_id, date_str in sorted_tasks:
        
        if user_id != current_user:
            print(f"\n👤 正在處理使用者: {user_id}")
            current_user = user_id
            
        # 呼叫處理函式
        total, skipped, processed = process_user_date(db, user_id, date_str, args.model, args)
        
        if total > 0:
            print(f"   📅 日期 {date_str}: 原始資料 {total} 筆 -> 新增建議 {processed} 筆 (略過 {skipped})")
            total_processed_global += processed
        elif args.verbose:
            print(f"   📅 日期 {date_str}: 無有效 records 資料")

    if total_processed_global == 0 and len(tasks) == 0:
        print("❌ 依然找不到資料。可能原因：")
        print("   1. 資料庫真的空的。")
        print("   2. 集合名稱不是 'health_data' (請檢查大小寫)。")

    print("-" * 60)
    print(f"✅ 任務完成！總計新增建議: {total_processed_global} 筆")

if __name__ == "__main__":
    main()
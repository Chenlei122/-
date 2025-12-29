# -*- coding: utf-8 -*-
import argparse
import datetime
import re
import sys
from typing import Dict, Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, RetryError

# ... (省略 generate_payload 和 to_float，保持與 V8 一致) ...

def process_user_date(db, user_id, date_str, model_path, args) -> tuple:
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    target_ref = db.collection("User_advice_results")

    try:
        # === V9 優化：增加 .limit() 限制單次讀取量，節省額度 ===
        # 且我們只撈取尚未處理過的資料 (如果 Firestore 索引已建立)
        # 這裡先使用最安全的 limit 方式
        query = source_ref.limit(args.limit)
        docs = list(query.stream(timeout=5))
            
    except (ResourceExhausted, RetryError):
        print(f"\n❌ [!炸了!] 額度已耗盡。這代表每 {args.interval} 分鐘跑一次對你的資料量來說太快了。")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        return 0, 0, 0

    if not docs:
        return 0, 0, 0

    batch = db.batch()
    count = 0
    skipped = 0
    processed = 0

    for doc in docs:
        data = doc.to_dict()
        # 只處理沒標記過的
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
        # ... (batch commit 邏輯不變) ...

    if count > 0 and not args.dry_run:
        batch.commit()

    return len(docs), skipped, processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    # === V9 關鍵參數：頻繁執行時，days 建議設為 0 (只檢查當天) ===
    parser.add_argument("--days", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=100, help="單次處理上限，防止額度瞬間爆掉")
    parser.add_argument("--interval", type=str, default="30m") # 僅供顯示參考
    
    # ... (其他參數不變) ...
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    # 邏輯：只針對目標日期敲門
    today = datetime.date.today()
    target_dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days + 1)]

    print(f"⚡ V9 增量模式：檢查過去 {args.days} 天，單次上限 {args.limit} 筆")
    
    total_new = 0
    for d_str in target_dates:
        _, skipped, processed = process_user_date(db, args.target_user, d_str, args.model, args)
        total_new += processed
        if processed > 0:
            print(f"✅ {d_str}: 新增 {processed} 筆建議")

    print(f"🏁 任務結束。新增總數: {total_new}")

if __name__ == "__main__":
    main()

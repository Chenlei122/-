# -*- coding: utf-8 -*-
import argparse
import datetime
import re
import sys
from typing import Dict, Any, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, RetryError

# ---------------------------------------------------------
# 1. AI 邏輯載入
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
    tips = ["數據顯示活動量偏低，起來走走吧 🚶"] if steps < 2000 else ["數據正常，繼續保持！"]
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
# 3. Firestore 核心邏輯
# ---------------------------------------------------------
def init_firestore(key_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def process_user_date(db, user_id, date_str, model_path, args) -> tuple:
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    target_ref = db.collection("User_advice_results")

    try:
        # V9 核心優化：限制單次抓取的筆數，且加入超時控制
        query = source_ref.limit(args.limit)
        docs = list(query.stream(timeout=10))
    except (ResourceExhausted, RetryError):
        print(f"\n❌ [Quota Exceeded] 額度已耗盡！請考慮降低執行頻率或調低 --limit。")
        sys.exit(1) # 強制停止 GitHub Action，避免無謂重試
    except Exception as e:
        print(f"\n❌ 讀取時發生錯誤: {e}")
        return 0, 0, 0

    if not docs:
        return 0, 0, 0

    batch = db.batch()
    count = 0
    processed = 0
    skipped = 0

    for doc in docs:
        data = doc.to_dict()
        # 增量邏輯：如果已經處理過且沒開啟 force 模式，就跳過
        if (not args.force) and (data.get("_advised") is True):
            skipped += 1
            continue

        payload = generate_payload(data, model_path, user_id)
        if not payload: continue

        # 組合 ID 避免重複：使用者_日期_原始文件ID
        new_doc_id = f"{user_id}_{date_str}_{doc.id}"
        new_doc_ref = target_ref.document(new_doc_id)

        if not args.dry_run:
            batch.set(new_doc_ref, payload, merge=True)
            batch.update(doc.reference, {"_advised": True})
        
        processed += 1
        count += 1

        # 每 400 筆寫入一次
        if count >= 400 and not args.dry_run:
            batch.commit()
            batch = db.batch()
            count = 0

    if count > 0 and not args.dry_run:
        batch.commit()

    return len(docs), skipped, processed

# ---------------------------------------------------------
# 4. 主程式門戶
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Firebase Key Path")
    parser.add_argument("--days", type=int, default=0, help="往回檢查的天數 (頻繁執行建議設為 0)")
    parser.add_argument("--limit", type=int, default=50, help="單次處理筆數上限")
    parser.add_argument("--target-user", type=str, default="j3tgphl0e9gj2hPRaT6aAOhH1922")
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 V9 (增量狙擊模式)")
    print("-" * 60)
    
    today = datetime.date.today()
    target_dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days + 1)]

    total_added = 0
    for d_str in target_dates:
        total, skipped, processed = process_user_date(db, args.target_user, d_str, args.model, args)
        total_added += processed
        if total > 0:
            print(f"📅 {d_str}: 掃描 {total} 筆 -> 新增 {processed} 筆 (略過 {skipped})")

    print("-" * 60)
    print(f"✅ 任務完成！共新增 {total_added} 筆建議。")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
import os
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
# 1. 嘗試載入外部 AI 腦袋
# ---------------------------------------------------------
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except ImportError:
    print("⚠️ 警告: 找不到 predict_and_advise.py，將使用內建簡易規則。")
    _HAVE_MAKE_ADVICE = False

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
    avg_hr = to_float(data.get("avg_heart_rate") or data.get("avg_hr"), 0)
    
    tips = []
    state = "ok"
    
    if steps < 2000:
        tips.append("數據顯示活動量偏低，起來走走吧 🚶")
        state = "under_recovered"
    elif steps > 15000:
        tips.append("哇！今天走超級多路，請務必好好休息 🛌")
        state = "overreached"
        
    if avg_hr > 100:
        tips.append("平均心跳偏高，深呼吸放鬆一下 💓")
    
    if not tips:
        tips.append("一切看起來都很正常，保持下去！✨")

    return {
        "advice": {
            "state": state,
            "tips": tips,
            "model_version": "fallback-system-v3",
            "confidence": 0.5
        },
        "_advised": True
    }

def generate_payload(data: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    if _HAVE_MAKE_ADVICE:
        try:
            return _external_make_advice(data, model_path=model_path)
        except Exception as e:
            print(f"  [Model Error] AI 分析失敗，轉為內建規則: {e}")
    return rule_based_fallback(data)

# ---------------------------------------------------------
# 3. Firestore 核心操作
# ---------------------------------------------------------
def init_firestore(key_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def get_date_range(days):
    base = datetime.date.today()
    return [(base - datetime.timedelta(days=i)).isoformat() for i in range(days)]

def batch_write_safe(batch, commit=True):
    if commit:
        try:
            batch.commit()
            return True
        except (ResourceExhausted, ServiceUnavailable):
            print("⚠️ 寫入忙碌，暫停 5 秒後重試...")
            time.sleep(5)
            batch.commit()
            return True
        except Exception as e:
            print(f"❌ 批次寫入失敗: {e}")
            return False
    return False

def process_user_date(db, user_id, date_str, model_path, args) -> tuple:
    # Path: users/{uid}/health_data/{date}/records
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    # Path: users/{uid}/advice_results
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
    BATCH_LIMIT = 400

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
        else:
            print(f"  [Dry-Run] Advice: {payload.get('advice', {}).get('tips')}")

        processed += 1
        count += 1

        if count >= BATCH_LIMIT and not args.dry_run:
            batch_write_safe(batch)
            batch = db.batch()
            count = 0

    if count > 0 and not args.dry_run:
        batch_write_safe(batch)

    return len(docs), skipped, processed

# ---------------------------------------------------------
# 4. 主程式入口
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
    print(f"🚀 AI 建議生成器 (超級偵探版)")
    print(f"🔗 連接專案 ID: {db.project}")  # <-- 關鍵檢查點
    print("="*60)

    # 1. 嘗試正規掃描
    print("🔍 掃描使用者清單 (方法 A: 讀取 users 集合)...")
    user_ids = []
    try:
        user_docs = list(db.collection("users").stream())
        user_ids = [u.id for u in user_docs]
    except Exception as e:
        print(f"⚠️ 方法 A 失敗: {e}")

    # 2. 如果沒找到，嘗試「幽靈探測」(Collection Group Query)
    if not user_ids:
        print("👻 方法 A 找不到人 (可能是幽靈文件)，切換為方法 B: 反向搜尋...")
        try:
            # 搜尋所有名為 'health_data' 的集合，反推它們的父母是誰
            # 這招可以抓到那些「沒有被明確建立」的使用者文件
            blobs = list(db.collection_group("health_data").limit(20).stream())
            found_set = set()
            for b in blobs:
                # b.reference.parent.parent 就是使用者 Document Reference
                parent_doc = b.reference.parent.parent
                if parent_doc:
                    found_set.add(parent_doc.id)
            user_ids = list(found_set)
        except Exception as e:
            print(f"❌ 方法 B 也失敗: {e}")

    if not user_ids:
        print("❌ 錯誤: 真的找不到任何使用者。")
        print("   1. 請檢查上方的「專案 ID」是否正確？")
        print("   2. 請確認資料庫裡真的有 'health_data' 集合嗎？")
        return

    print(f"👥 確認鎖定 {len(user_ids)} 位使用者: {user_ids}")
    
    start_time = time.time()
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
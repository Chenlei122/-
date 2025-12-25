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
# 1. 嘗試載入外部 AI 腦袋 (predict_and_advise.py)
# ---------------------------------------------------------
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except ImportError:
    print("⚠️ 警告: 找不到 predict_and_advise.py，將使用內建簡易規則。")
    _HAVE_MAKE_ADVICE = False

# ---------------------------------------------------------
# 2. 輔助工具 & 內建備用規則
# ---------------------------------------------------------
_NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")

def to_float(x, d: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        m = _NUMERIC_LIKE.search(str(x))
        return float(m.group()) if m else d
    except Exception: return d

def rule_based_fallback(data: Dict[str, Any]) -> Dict[str, Any]:
    """當 AI 模型失效時的備用邏輯"""
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
# 3. Firestore 核心操作 (支援 User-Centric 結構)
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
    """
    處理單一使用者在特定日期的資料
    路徑: users/{uid}/health_data/{date}/records
    輸出: users/{uid}/advice_results/{date_recordID}
    """
    # 1. 設定讀取路徑 (對應新的資料結構)
    # users -> {uid} -> health_data -> {date} -> records
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    # 2. 設定寫入路徑 (建議放在使用者的資料夾內，比較整齊)
    # users -> {uid} -> advice_results
    target_ref = db.collection("users").document(user_id).collection("advice_results")

    try:
        # 使用 stream() 讀取資料
        docs = list(source_ref.stream())
    except Exception:
        # 如果該日期沒有資料，直接回傳 0
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

        # 檢查是否處理過 (除非開啟 --force)
        if (not args.force) and (data.get("_advised") is True):
            skipped += 1
            continue

        # 產生建議
        payload = generate_payload(data, model_path)
        if not payload: continue

        # 設定新文件的 ID：日期_原始ID
        new_doc_id = f"{date_str}_{doc.id}"
        new_doc_ref = target_ref.document(new_doc_id)

        # 寫入 Advice
        if not args.dry_run:
            batch.set(new_doc_ref, payload, merge=True)
            # 標記原資料已處理
            batch.update(doc.reference, {"_advised": True})
        else:
            print(f"  [Dry-Run] User: {user_id} | ID: {new_doc_id} | State: {payload.get('advice', {}).get('state')}")

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
    parser = argparse.ArgumentParser(description="Firestore AI Advice (Multi-User Support)")
    parser.add_argument("--key", required=True, help="Service Account JSON")
    parser.add_argument("--days", type=int, default=7, help="回看天數")
    parser.add_argument("--model", default="behavior_health_model.joblib", help="模型路徑")
    parser.add_argument("--force", action="store_true", help="強制重跑")
    parser.add_argument("--dry-run", action="store_true", help="模擬執行")
    parser.add_argument("--verbose", action="store_true", help="詳細日誌")
    # 不需要 --root 參數了，因為我們現在固定抓 users

    args = parser.parse_args()
    db = init_firestore(args.key)
    dates = get_date_range(args.days)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (多用戶版)")
    print(f"📅 掃描日期: 最近 {args.days} 天")
    print("="*60)

    # 1. 取得所有使用者 ID
    print("🔍 正在掃描使用者清單...")
    users_ref = db.collection("users")
    try:
        user_docs = list(users_ref.stream())
        user_ids = [u.id for u in user_docs]
    except Exception as e:
        print(f"❌ 無法讀取使用者清單: {e}")
        return

    print(f"👥 找到 {len(user_ids)} 位使用者: {user_ids}")
    
    start_time = time.time()
    total_processed_global = 0

    # 2. 針對「每一位使用者」進行處理
    for uid in user_ids:
        print(f"\n👤 正在處理使用者: {uid}")
        
        # 針對該使用者的「每一天」
        for d in tqdm(dates, desc=f"   Checking days for {uid[:5]}..."):
            scanned, skip, proc = process_user_date(db, uid, d, args.model, args)
            total_processed_global += proc
            
            if args.verbose and proc > 0:
                tqdm.write(f"   -> {d}: 新增 {proc} 筆")

    end_time = time.time()
    print("-" * 60)
    print(f"[執行完畢] 耗時 {end_time - start_time:.2f} 秒")
    print(f"✅ 總共為所有使用者新增建議: {total_processed_global} 筆")
    print("-" * 60)

if __name__ == "__main__":
    main()
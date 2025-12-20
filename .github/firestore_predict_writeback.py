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
# 2. 輔助工具 & 內建備用規則 (來自第一份代碼，以防模型失效)
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
    """統一產出建議 Payload (優先用外部模型，失敗則用內建規則)"""
    if _HAVE_MAKE_ADVICE:
        try:
            return _external_make_advice(data, model_path=model_path)
        except Exception as e:
            print(f"  [Model Error] AI 分析失敗，轉為內建規則: {e}")
    
    return rule_based_fallback(data)

# ---------------------------------------------------------
# 3. Firestore 核心操作 (Sidecar 模式)
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
    """安全寫入，處理 Google 限制"""
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

def process_date_collection(db, root_name, date_str, model_path, args) -> tuple:
    """
    核心邏輯：
    1. 讀取 records/{date}/records
    2. 寫入 advice_results/{date_id}
    3. 更新 records/{date}/records/{id} (_advised=True)
    """
    source_ref = db.collection(root_name).document(date_str).collection("records")
    target_ref = db.collection("advice_results") # 新的存放區

    try:
        docs = list(source_ref.stream())
    except Exception:
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
        
        # 如果 Payload 為空 (極少見)，跳過
        if not payload:
            continue

        # 設定新文件的 ID：日期_原始ID (例如 2025-03-03_20-44-23)
        new_doc_id = f"{date_str}_{doc.id}"
        new_doc_ref = target_ref.document(new_doc_id)

        # 動作 1: 寫入新區域 (advice_results)
        # 注意：我們把 payload 寫進去，這包含了 advice 和 location 等資訊
        if not args.dry_run:
            batch.set(new_doc_ref, payload, merge=True)

        # 動作 2: 標記舊區域 (records)
        # 這樣下次就不會重複跑
        if not args.dry_run:
            batch.update(doc.reference, {"_advised": True})
        else:
            # Dry-run 模式下只印不寫
            print(f"  [Dry-Run] 將寫入: {new_doc_id} | 狀態: {payload.get('advice', {}).get('state')}")

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
    parser = argparse.ArgumentParser(description="Firestore AI Advice Generator (Sidecar Pattern)")
    parser.add_argument("--key", required=True, help="Service Account JSON 路徑")
    parser.add_argument("--root", default="records", help="原始資料集合名稱 (預設: records)")
    parser.add_argument("--days", type=int, default=30, help="回看幾天 (預設: 30)")
    parser.add_argument("--model", default="behavior_health_model.joblib", help="模型路徑")
    parser.add_argument("--force", action="store_true", help="強制重跑 (忽略 _advised 標記)")
    parser.add_argument("--dry-run", action="store_true", help="模擬執行 (不寫入資料庫)")
    parser.add_argument("--verbose", action="store_true", help="顯示詳細日誌")

    args = parser.parse_args()

    # 初始化
    db = init_firestore(args.key)
    dates = get_date_range(args.days)
    
    print("="*60)
    print(f"🚀 AI 建議生成器啟動 (v3.0-Sidecar)")
    print(f"📂 讀取來源: {args.root}/[日期]/records")
    print(f"💾 寫入目標: advice_results (獨立集合)")
    print(f"📅 掃描範圍: 最近 {args.days} 天")
    if args.force: print("⚠️  模式: 強制覆寫 (Force Mode)")
    if args.dry_run: print("🧪 模式: 模擬測試 (Dry Run)")
    print("="*60)

    total_scanned = 0
    total_skipped = 0
    total_processed = 0
    start_time = time.time()

    # 使用 tqdm 顯示進度條
    for d in tqdm(dates, desc="[處理進度]"):
        scanned, skip, proc = process_date_collection(db, args.root, d, args.model, args)
        total_scanned += scanned
        total_skipped += skip
        total_processed += proc
        
        if args.verbose and proc > 0:
            tqdm.write(f"  -> {d}: 新增 {proc} 筆建議")

    end_time = time.time()
    print("-" * 60)
    print(f"[執行完畢] 耗時 {end_time - start_time:.2f} 秒")
    print(f"📊 總掃描文件: {total_scanned}")
    print(f"⏭️  跳過 (已處理): {total_skipped}")
    print(f"✅ 新增建議 (Advice): {total_processed}")
    print("-" * 60)

if __name__ == "__main__":
    main()
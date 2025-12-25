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
# 4. 主程式 (鎖定使用者 + 自動掃描所有日期)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    
    # [修正點] 把 --days 加回來，防止舊指令報錯 (雖然程式現在會自動掃全日期)
    parser.add_argument("--days", type=int, default=7, help="(已棄用) 相容性保留參數")

    # 預設使用者 ID (你提供的那個)，如果不帶參數就會用這個
    default_user = "j3tgphl0e9gj2hPRaT6aAOhH1922"
    
    parser.add_argument("--target-user", type=str, default=default_user, help="指定要掃描的使用者 ID")
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (自動日期掃描版)")
    print("="*60)
    print(f"👤 目標使用者: {args.target_user}")
    print(f"🔍 正在搜尋該使用者 `health_data` 底下的所有日期...")

    # 1. 直接定位到該使用者的 health_data 資料夾
    health_ref = db.collection("users").document(args.target_user).collection("health_data")

    try:
        # 2. 自動抓取所有「日期文件」
        date_docs = list(health_ref.stream())
    except Exception as e:
        print(f"❌ 讀取日期失敗: {e}")
        return

    if not date_docs:
        print("❌ 找不到任何日期資料夾！")
        print(f"   請確認路徑: users/{args.target_user}/health_data/ 是否存在")
        return

    print(f"📊 發現 {len(date_docs)} 個日期資料夾，準備開始處理...")
    print("-" * 60)

    total_processed_global = 0

    # 3. 自動迴圈處理每一個日期
    sorted_dates = sorted([doc.id for doc in date_docs])

    for date_str in sorted_dates:
        # 呼叫處理函式
        total, skipped, processed = process_user_date(
            db, args.target_user, date_str, args.model, args
        )

        if total > 0:
            print(f"📅 {date_str}: 發現 {total} 筆紀錄 -> ✅ 新增 {processed} 筆建議 (略過 {skipped})")
            total_processed_global += processed
        else:
            if args.verbose:
                print(f"📅 {date_str}: (無紀錄)")

    print("-" * 60)
    print(f"✅ 全部完成！總計對 {len(date_docs)} 個日期進行檢查，新增 {total_processed_global} 筆建議。")

if __name__ == "__main__":
    main()
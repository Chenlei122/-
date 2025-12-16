# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import re
import sys
import time
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta, timezone

from google.cloud import firestore
from google.oauth2 import service_account

# 版本號：增量處理專用版
VER = "v3.0-incremental"

# --- 嘗試載入外部 advice 模組 (你的預測大腦) ---
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except Exception:
    _HAVE_MAKE_ADVICE = False

# ---------------------------
# 基礎工具 (保持不變)
# ---------------------------
_NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")

def to_int(x, d: Optional[int] = None) -> Optional[int]:
    try:
        if isinstance(x, (int, float)): return int(x)
        m = _NUMERIC_LIKE.search(str(x))
        return int(float(m.group())) if m else d
    except Exception: return d

def to_float(x, d: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        m = _NUMERIC_LIKE.search(str(x))
        return float(m.group()) if m else d
    except Exception: return d

def parse_weather(x):
    cond, temp_c = None, None
    if isinstance(x, dict):
        cond = x.get("label") or x.get("condition")
        temp_c = to_float(x.get("temp_c"))
    elif isinstance(x, str):
        parts = [p.strip() for p in x.split(",")]
        if parts: cond = parts[0] or None
        m = re.search(r"([-+]?\d+)\s*°?C", x, flags=re.IGNORECASE)
        if m: temp_c = to_float(m.group(1))
    elif isinstance(x, (int, float)):
        temp_c = float(x)
    return cond, temp_c

def build_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if "steps" in raw: f["steps"] = to_int(raw.get("steps"))
    for k in ["avgheartrate", "avg_heart_rate", "heartrate", "heart_rate"]:
        if k in raw:
            f["avg_heart_rate"] = to_float(raw.get(k)); break
    for k in ["maxheartrate", "max_heart_rate"]:
        if k in raw:
            f["max_heart_rate"] = to_float(raw.get(k)); break
    
    cond, temp_c = None, None
    if "weather" in raw: cond, temp_c = parse_weather(raw.get("weather"))
    if "condition" in raw: cond = cond or raw.get("condition")
    if temp_c is None: temp_c = to_float(raw.get("temp_c"))

    if cond: f["condition"] = cond
    if temp_c is not None: f["temp_c"] = temp_c
    return f

# ---------------------------
# 內建保守規則
# ---------------------------
def rule_based_state(feat: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    steps = feat.get("steps")
    avg_hr = feat.get("avg_heart_rate")
    state = "ok"
    reasons: List[str] = []

    if steps is not None:
        if steps < 2000: state = "under_recovered"; reasons.append(f"low steps ({steps})")
        elif steps > 12000: state = "overreached"; reasons.append(f"high steps ({steps})")
    
    if avg_hr is not None and avg_hr > 100:
        state = "under_recovered"; reasons.append(f"high avg_hr ({avg_hr})")
    
    tips: List[str] = []
    if state == "under_recovered":
        tips = ["今天請放慢節奏，多休息補水。"]
    elif state == "overreached":
        tips = ["運動量很大，明天建議安排恢復日。"]
    else:
        tips = ["狀態不錯，維持目前節奏即可。"]
    return state, tips, reasons

# ---------------------------
# 核心邏輯 (增量處理)
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--key", help="service account json path")
    p.add_argument("--root", default="health_data")
    p.add_argument("--days", type=int, default=3, help="回看幾天內的資料夾")
    p.add_argument("--force", action="store_true", help="【強制重算】忽略已處理的記號")
    p.add_argument("--dry-run", action="store_true", help="只顯示不寫入")
    p.add_argument("--verbose", action="store_true", help="顯示詳細處理過程")
    p.add_argument("--write-subdoc", action="store_true")
    p.add_argument("--since-hours", type=int, default=None) 
    return p.parse_args()

def main():
    start_time = time.time()
    args = parse_args()

    # 1. 初始化 Firestore
    if args.key:
        creds = service_account.Credentials.from_service_account_file(args.key)
        db = firestore.Client(credentials=creds, project=creds.project_id)
    else:
        db = firestore.Client()
    
    print(f"[{VER}] 專案: {db.project} | 模式: {'強制重跑' if args.force else '增量更新 (只跑新資料)'}")

    # 2. 鎖定日期範圍 (只檢查最近 N 天，避免掃描整年資料)
    now_utc = datetime.now(timezone.utc)
    target_days = args.days
    if args.since_hours:
        target_days = max(1, int(args.since_hours / 24) + 1)
    
    target_date_ids = []
    for i in range(target_days + 1): 
        d = now_utc - timedelta(days=i)
        target_date_ids.append(d.strftime("%Y-%m-%d"))
    # 相容舊格式
    target_date_ids += [d.replace("-", "/") for d in target_date_ids]
    
    print(f"[{VER}] 掃描日期清單: {target_date_ids}")

    batch = db.batch()
    batch_count = 0
    stats = {"scanned": 0, "skipped": 0, "processed": 0, "errors": 0}

    root_ref = db.collection(args.root)

    # 3. 逐日處理
    for date_id in target_date_ids:
        records_ref = root_ref.document(date_id).collection("records")
        
        # [優化] 使用 stream() 一次抓回資料，減少網路來回次數
        # 如果不是強制重跑，我們可以在 Python 端快速過濾
        try:
            docs_stream = records_ref.stream()
        except Exception:
            # 該日期資料夾可能不存在，直接跳過
            continue

        date_processed_count = 0
        
        for doc in docs_stream:
            stats["scanned"] += 1
            data = doc.to_dict() or {}
            
            # ==========================================
            # 🛑 核心需求 1 & 2：檢查記號與跳過
            # ==========================================
            # 如果沒有 --force 參數，且資料內已有 '_advised' 記號
            if (not args.force) and data.get("_advised") is True:
                stats["skipped"] += 1
                if args.verbose:
                    print(f"  [SKIP] {doc.id} (已處理過)")
                continue  # 直接從這裡跳過，繼續下一筆
            
            # 程式跑到這裡，代表是「沒記號」的新資料
            # ==========================================

            # 4. 開始分析
            advice_payload = None
            
            # (A) 呼叫你的 AI 模型 (predict_and_advise.py)
            if _HAVE_MAKE_ADVICE:
                try:
                    advice_payload = _external_make_advice(data, model_path="behavior_health_model.joblib")
                except Exception as e:
                    if args.verbose: print(f"    [Model Error] {e}")
                    stats["errors"] += 1
            
            # (B) 如果模型失敗，用規則補救
            if not advice_payload:
                feat = build_features(data)
                if not feat: 
                    # 資料缺損太嚴重，沒救
                    continue 
                state, tips, reasons = rule_based_state(feat)
                advice_payload = {
                    "advice": {
                        "state": state, "tips": tips, "_reason": reasons, 
                        "model_version": "fallback-v3"
                    },
                    "_advised": True
                }

            if not advice_payload: continue

            # 5. 加上記號 (Marking)
            advice_payload["_advised"] = True
            advice_payload["_advised_at_utc"] = firestore.SERVER_TIMESTAMP
            advice_payload["_version"] = VER

            # 6. 寫入批次
            if args.dry_run:
                print(f"[DRY-RUN] 會寫入 -> {doc.id} : {advice_payload.get('advice', {}).get('state')}")
            else:
                batch.set(doc.reference, advice_payload, merge=True)
                
                # 同步寫入子文件 (舊相容)
                if args.write_subdoc:
                    sub_ref = doc.reference.collection("advice").document("v1")
                    batch.set(sub_ref, advice_payload.get("advice", {}), merge=True)

                batch_count += 1
                stats["processed"] += 1
                date_processed_count += 1

                # 每 400 筆上傳一次 (Firebase 限制 500)
                if batch_count >= 400:
                    batch.commit()
                    print(f"[{VER}] ⚡ 批次上傳成功 (400 筆)")
                    batch = db.batch()
                    batch_count = 0
        
        if args.verbose and date_processed_count > 0:
            print(f" -> 日期 {date_id}: 新增處理了 {date_processed_count} 筆")

    # 7. 收尾：上傳剩下的
    if batch_count > 0 and not args.dry_run:
        batch.commit()
        print(f"[{VER}] ⚡ 最後批次上傳成功 ({batch_count} 筆)")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[{VER}] 執行完畢 (耗時 {elapsed:.2f} 秒)")
    print(f"📊 總掃描: {stats['scanned']} | ⏭️  跳過(已看過): {stats['skipped']} | ✅ 新處理: {stats['processed']}")
    print("-" * 50)

if __name__ == "__main__":
    main()
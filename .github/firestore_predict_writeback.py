# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import re
import sys
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta, timezone

from google.cloud import firestore
from google.oauth2 import service_account

VER = "v1.4-hardened"

# --- 可選：嘗試載入你前面完成的韌性版建議函式 ---
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except Exception as _e:
    _HAVE_MAKE_ADVICE = False

# ---------------------------
# 基礎工具
# ---------------------------
_NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")

def to_int(x, d: Optional[int] = None) -> Optional[int]:
    try:
        if isinstance(x, (int, float)): return int(x)
        m = _NUMERIC_LIKE.search(str(x))
        return int(float(m.group())) if m else d
    except Exception:
        return d

def to_float(x, d: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        m = _NUMERIC_LIKE.search(str(x))
        return float(m.group()) if m else d
    except Exception:
        return d

def parse_weather(x):
    """
    支援：
      - dict: {"condition": "...", "temp_c": 24} 或韌性版 {"label": "...", "temp_c": 24}
      - str : "Clouds, 24°C" / "晴 27C" / "Clouds"
      - num : 視為 temp_c
    回傳: (condition_label, temp_c)
    """
    cond, temp_c = None, None
    if isinstance(x, dict):
        cond = x.get("label") or x.get("condition")
        temp_c = to_float(x.get("temp_c"))
    elif isinstance(x, str):
        # "Clouds, 24°C" -> ("Clouds", 24)
        parts = [p.strip() for p in x.split(",")]
        if parts:
            cond = parts[0] or None
        m = re.search(r"([-+]?\d+)\s*°?C", x, flags=re.IGNORECASE)
        if m:
            temp_c = to_float(m.group(1))
    elif isinstance(x, (int, float)):
        temp_c = float(x)
    return cond, temp_c

def build_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    把多種命名/格式規範成固定 features，供簡單規則用。
    （如果你外部 make_advice 有更完整的抽取，這裡只是 fallback）
    """
    f: Dict[str, Any] = {}
    # steps
    if "steps" in raw: f["steps"] = to_int(raw.get("steps"))

    # avg heart rate
    for k in ["avgheartrate", "avg_heart_rate", "heartrate", "average_heart_rate", "heart_rate"]:
        if k in raw: 
            f["avg_heart_rate"] = to_float(raw.get(k)); 
            break

    # max heart rate
    for k in ["maxheartrate", "max_heart_rate"]:
        if k in raw: 
            f["max_heart_rate"] = to_float(raw.get(k)); 
            break

    # weather
    cond, temp_c = None, None
    if "weather" in raw:
        cond, temp_c = parse_weather(raw.get("weather"))
    if "condition" in raw or "temp_c" in raw:
        cond = cond or raw.get("condition")
        if temp_c is None: 
            temp_c = to_float(raw.get("temp_c"))

    if cond: f["condition"] = cond
    if temp_c is not None: f["temp_c"] = temp_c
    return f

# ---------------------------
# 內建保守規則（模型/外部失敗時用）
# ---------------------------
def rule_based_state(feat: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    steps = feat.get("steps")
    avg_hr = feat.get("avg_heart_rate")
    max_hr = feat.get("max_heart_rate")
    temp_c = feat.get("temp_c")

    state = "ok"
    reasons: List[str] = []

    if steps is not None:
        if steps < 2000:
            state = "under_recovered"; reasons.append(f"very low steps ({steps})")
        elif steps > 12000:
            state = "overreached"; reasons.append(f"very high steps ({steps})")
        else:
            reasons.append(f"steps in range ({steps})")
    else:
        reasons.append("no steps -> default ok")

    if avg_hr is not None and avg_hr > 100:
        state = "under_recovered"; reasons.append(f"high avg_hr ({avg_hr})")
    if max_hr is not None and max_hr > 170:
        state = "overreached"; reasons.append(f"very high max_hr ({max_hr})")
    if temp_c is not None and temp_c >= 32 and state == "ok":
        state = "under_recovered"; reasons.append(f"hot ({temp_c}C) -> recover")

    tips: List[str] = []
    if state == "under_recovered":
        tips += [
            "今天放慢節奏，做 20–30 分鐘輕鬆步行或伸展。",
            "補 300–500ml 水分，優先早睡 30 分鐘。"
        ]
    elif state == "overreached":
        tips += [
            "已經很努力！明天安排低強度日，避免連續高負荷。",
            "做 5–10 分鐘呼吸放鬆，補充電解質與蛋白質。"
        ]
    else:
        tips += [
            "保持現在的節奏，維持 30–45 分鐘中等強度活動。",
            "外出活動記得補水、防曬、防中暑。"
        ]
    return state, tips, reasons

# ---------------------------
# Firestore 寫入
# ---------------------------
def write_advice(doc_ref, advice: Dict[str, Any], write_subdoc: bool, dry=False):
    """
    advice: 須是「advice.*」內容本體（不含 _advised 等）
    """
    payload = {
        "advice": advice,
        "_advised": True,
        "_advised_at_utc": firestore.SERVER_TIMESTAMP,
        "_version": VER,
    }
    if dry:
        print(f"[DRY] would write -> {doc_ref.path} :: {payload}")
        return
    doc_ref.set(payload, merge=True)
    if write_subdoc:
        doc_ref.collection("advice").document("v1").set(advice, merge=True)

# ---------------------------
# 主流程
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--key", help="service account json path")
    p.add_argument("--root", default="health_data")
    p.add_argument("--days", type=int, default=7, help="歷史回看天數（和 since-hours 擇一）")
    p.add_argument("--since-hours", type=int, default=None, help="回看近幾小時，優先於 days")
    p.add_argument("--force", action="store_true", help="即使 _advised 已存在也覆寫")
    p.add_argument("--dry-run", action="store_true", help="不寫入，只列印")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--write-subdoc", action="store_true", help="同步寫到 advice/v1 子文件")
    return p.parse_args()

def banner(db: firestore.Client, args):
    print(f"[{VER}] Firestore project = {db.project}")
    print(f"[{VER}] root={args.root} days={args.days} since_hours={args.since_hours} force={args.force} dry_run={args.dry_run} verbose={args.verbose}")

def main():
    args = parse_args()

    # 綁定專案
    if args.key:
        creds = service_account.Credentials.from_service_account_file(args.key)
        db = firestore.Client(credentials=creds, project=creds.project_id)
    else:
        db = firestore.Client()

    banner(db, args)

    now = datetime.now(timezone.utc)
    if args.since_hours and args.since_hours > 0:
        start_dt = now - timedelta(hours=args.since_hours)
    else:
        start_dt = now - timedelta(days=args.days)

    root = db.collection(args.root)

    # 日期 doc 可能不只 YYYY-MM-DD；做穩健 parse + 排序
    date_docs = list(root.list_documents())
    def _parse_date_id(did: str) -> Optional[datetime]:
        # 嘗試以 YYYY-MM-DD 解析
        try:
            return datetime.strptime(did, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    date_pairs = []
    for dd in date_docs:
        ts = _parse_date_id(dd.id)
        if ts is not None and ts >= start_dt:
            date_pairs.append((ts, dd))
    date_pairs.sort()  # 由舊到新

    seen = wrote = 0
    for ts, date_doc in date_pairs:
        date_id = date_doc.id
        print(f"[{VER}] DATE {date_id}")

        # 時間 doc 排序（假設 HH:MM:SS）
        recs = list(date_doc.collection("records").list_documents())
        recs.sort(key=lambda r: r.id)  # 由早到晚
        if not recs:
            print(f"[{VER}]  - no records")
            continue

        for r in recs:
            snap = r.get()
            if not snap.exists:
                continue
            data = snap.to_dict() or {}
            seen += 1

            if args.verbose:
                print(f"[{VER}]   REC {r.id} raw_keys={sorted(list(data.keys()))}")

            if (not args.force) and data.get("_advised"):
                if args.verbose:
                    print(f"[{VER}]    SKIP: already _advised (use --force to overwrite)")
                continue

            # 決定是否有「可用特徵」
            has_useful = any(k in data for k in ("steps", "avgheartrate", "avg_heart_rate", "heartrate", "weather", "condition", "temp_c"))
            if not has_useful:
                # 寫入一個 skip 訊息幫助你在 Console 看到原因
                if args.verbose:
                    print(f"[{VER}]    SKIP: no usable features")
                if not args.dry_run:
                    r.set({"_advice_skip": "no_usable_features"}, merge=True)
                continue

            # 先試你外部的韌性 make_advice（含模型/規則），若沒有則用內建 rule-based
            advice_block: Dict[str, Any]
            if _HAVE_MAKE_ADVICE:
                try:
                    # 外部 make_advice 會直接回傳包含 advice.* 的結構或 payload
                    payload = _external_make_advice(data, model_path="behavior_health_model.joblib")
                    # 兼容：若外部回傳已含 "advice" 就直接取；否則視為 advice 本體
                    advice_block = payload.get("advice", payload)
                except Exception as e:
                    if args.verbose:
                        print(f"[{VER}]    external make_advice failed: {e} -> fallback rules")
                    feat = build_features(data)
                    state, tips, reasons = rule_based_state(feat)
                    advice_block = {
                        "state": state,
                        "tips": tips,
                        "features_used": feat,
                        "missing_features": [k for k in ["steps","avg_heart_rate","max_heart_rate","temp_c"] if k not in feat],
                        "model_version": "fallback-v1",
                        "_reason": reasons
                    }
            else:
                feat = build_features(data)
                state, tips, reasons = rule_based_state(feat)
                advice_block = {
                    "state": state,
                    "tips": tips,
                    "features_used": feat,
                    "missing_features": [k for k in ["steps","avg_heart_rate","max_heart_rate","temp_c"] if k not in feat],
                    "model_version": "fallback-v1",
                    "_reason": reasons
                }

            try:
                write_advice(r, advice_block, write_subdoc=args.write_subdoc, dry=args.dry_run)
                wrote += 1
                if args.verbose:
                    print(f"[{VER}]    WRITE ok -> {advice_block.get('state')}  used={advice_block.get('features_used')}")
            except Exception as e:
                print(f"[{VER}]    ERROR WRITE: {e}", file=sys.stderr)

    print(f"[{VER}] DONE seen={seen} wrote={wrote} (force={args.force} dry={args.dry_run})")
    # 若你希望 Actions 在 0 寫入時標紅，取消下面註解：
    # if wrote == 0:
    #     sys.exit(2)

if __name__ == "__main__":
    main()

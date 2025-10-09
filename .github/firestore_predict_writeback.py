# -*- coding: utf-8 -*-
import argparse, re, sys, math, datetime as dt
from typing import Dict, Any, Tuple, List

from google.cloud import firestore
from google.oauth2 import service_account

VER = "v1.3-debug-force"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=False, help="service account json path")
    p.add_argument("--root", default="health_data")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--force", action="store_true", help="write even if _advised=True")
    p.add_argument("--dry-run", action="store_true", help="do not write, only log")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def banner(db: firestore.Client, args):
    print(f"[{VER}] Firestore project = {db.project}")
    print(f"[{VER}] root={args.root} days={args.days} force={args.force} dry_run={args.dry_run} verbose={args.verbose}")

def parse_weather(x):
    cond, temp_c = None, None
    if isinstance(x, dict):
        cond = x.get("condition")
        temp_c = x.get("temp_c")
    elif isinstance(x, str):
        m = re.match(r"\s*([A-Za-z]+)\s*,\s*([\-]?\d+)\s*°?C\s*", x)
        if m:
            cond, temp_c = m.group(1), float(m.group(2))
        else:
            cond = x
    elif isinstance(x, (int, float)):
        temp_c = float(x)
    return cond, temp_c

def to_int(x, d=None):
    try: return int(float(x))
    except: return d

def to_float(x, d=None):
    try: return float(x)
    except: return d

def build_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    f = {}
    if "steps" in raw: f["steps"] = to_int(raw.get("steps"))
    for k in ["avgheartrate","avg_heart_rate","heartrate"]:
        if k in raw: f["avg_heart_rate"] = to_float(raw.get(k)); break
    for k in ["maxheartrate","max_heart_rate"]:
        if k in raw: f["max_heart_rate"] = to_float(raw.get(k)); break

    cond, temp_c = None, None
    if "weather" in raw:
        cond, temp_c = parse_weather(raw.get("weather"))
    if "condition" in raw or "temp_c" in raw:
        cond = cond or raw.get("condition")
        if temp_c is None: temp_c = to_float(raw.get("temp_c"))
    if cond: f["condition"] = cond
    if temp_c is not None: f["temp_c"] = temp_c
    return f

def rule_based_state(feat: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    steps = feat.get("steps")
    avg_hr = feat.get("avg_heart_rate")
    max_hr = feat.get("max_heart_rate")
    temp_c = feat.get("temp_c")
    reasons = []
    if steps is not None:
        if steps < 2000:
            state = "under_recovered"; reasons.append(f"very low steps ({steps})")
        elif steps > 12000:
            state = "overreached"; reasons.append(f"very high steps ({steps})")
        else:
            state = "ok"; reasons.append(f"steps in range ({steps})")
    else:
        state = "ok"; reasons.append("no steps -> default ok")
    if avg_hr is not None and avg_hr > 90:
        state = "under_recovered"; reasons.append(f"high avg_hr ({avg_hr})")
    if max_hr is not None and max_hr > 170:
        state = "overreached"; reasons.append(f"very high max_hr ({max_hr})")
    if temp_c is not None and temp_c >= 32 and state == "ok":
        state = "under_recovered"; reasons.append(f"hot ({temp_c}C) -> recover")

    tips = []
    if state == "under_recovered":
        tips += ["今天放慢節奏，做 20–30 分鐘輕鬆步行或伸展。","補 300–500ml 水分，優先早睡 30 分鐘。"]
    elif state == "overreached":
        tips += ["已經很努力！明天安排低強度日，避免連續高負荷。","做 5–10 分鐘呼吸放鬆，補充電解質與蛋白質。"]
    else:
        tips += ["保持現在的節奏，維持 30–45 分鐘中等強度活動。","外出活動記得補水、防曬、防中暑。"]
    return state, tips, reasons

def write_advice(doc_ref, advice: Dict[str, Any], dry=False):
    if dry:
        print(f"DRY-WRITE to {doc_ref.path}: {advice}")
        return
    doc_ref.set({"advice": advice, "_advised": True, "_advised_at_utc": firestore.SERVER_TIMESTAMP}, merge=True)
    doc_ref.collection("advice").document("v1").set(advice, merge=True)

def main():
    args = parse_args()

    # 用 key 明確綁定專案（避免默默用到別的 ADC）
    if args.key:
        creds = service_account.Credentials.from_service_account_file(args.key)
        db = firestore.Client(credentials=creds, project=creds.project_id)
    else:
        db = firestore.Client()

    banner(db, args)

    now = dt.datetime.utcnow()
    start_date = (now - dt.timedelta(days=args.days)).date().isoformat()
    end_date   = now.date().isoformat()

    root = db.collection(args.root)
    wrote = seen = 0
    print(f"[{VER}] scanning dates {start_date}..{end_date}")

    for date_doc in root.list_documents():
        date_id = date_doc.id
        if not (start_date <= date_id <= end_date):
            continue
        print(f"[{VER}] DATE {date_id}")
        recs = list(date_doc.collection("records").list_documents())
        if not recs:
            print(f"[{VER}]  - no records")
        for r in recs:
            seen += 1
            snap = r.get()
            data = snap.to_dict() or {}
            print(f"[{VER}]   REC {r.id} raw_keys={list(data.keys())}")

            if (not args.force) and data.get("_advised"):
                print(f"[{VER}]    SKIP: already _advised")
                continue

            feat = build_features(data)
            missing = [k for k in ["steps","avg_heart_rate","max_heart_rate","temp_c"] if k not in feat]
            print(f"[{VER}]    features={feat} missing={missing}")

            state, tips, reasons = rule_based_state(feat)
            advice = {
                "state": state,
                "tips": tips,
                "features_used": feat,
                "missing_features": missing,
                "_reason": reasons,
                "_advised": True,
                "_advised_at_utc": firestore.SERVER_TIMESTAMP,
                "_version": VER,
            }
            try:
                write_advice(r, advice, dry=args.dry_run)
                wrote += 1
                print(f"[{VER}]    WRITE -> {state}  reason={reasons}")
            except Exception as e:
                print(f"[{VER}]    ERROR WRITE: {e}", file=sys.stderr)

    print(f"[{VER}] DONE seen={seen} wrote={wrote} (force={args.force} dry={args.dry_run})")

if __name__ == "__main__":
    main()

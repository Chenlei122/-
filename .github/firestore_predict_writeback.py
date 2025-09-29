# firestore_predict_writeback.py
import re
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from predict_and_advise import predict_and_advise  # 會載用你的模型與建議
import argparse

REQUIRED = ["latitude","longitude","temp_c","condition","steps",
            "avg_heart_rate","max_heart_rate","sleep_hours",
            "stress_level","activity_minutes","is_raining","commuting_mode"]

def parse_weather(raw):
    """把 'Clouds, 24°C' 這種字串拆成 temp_c / condition / is_raining"""
    temp, cond, is_rain = None, None, 0
    if isinstance(raw, str):
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*°C", raw)
        if m:
            temp = float(m.group(1))
        cond = raw.split(",")[0].strip()
        if any(k in cond.lower() for k in ["rain", "drizzle", "thunder"]):
            is_rain = 1
    return temp, cond, is_rain

def to_features(doc_dict):
    """把 Firestore 的一筆資料轉成模型需要的特徵，若缺少就盡量從 weather 補。"""
    d = dict(doc_dict or {})
    out = {k: d.get(k) for k in REQUIRED if k in d and d.get(k) is not None}

    # 若沒 temp_c/condition/is_raining 就從 weather 補
    if "temp_c" not in out or "condition" not in out or "is_raining" not in out:
        t, c, ir = parse_weather(d.get("weather"))
        if "temp_c" not in out and t is not None:
            out["temp_c"] = t
        if "condition" not in out and c:
            out["condition"] = c
        if "is_raining" not in out:
            out["is_raining"] = ir

    # 合理的預設值（盡量避免因缺值而跳過）
    out.setdefault("condition", "Clouds")
    out.setdefault("is_raining", 0)
    out.setdefault("commuting_mode", d.get("commuting_mode", "walk"))

    missing = [k for k in REQUIRED if k not in out]
    return out, missing

def run(key_path, root="health_data", days=None, force=False, model_path="behavior_health_model.joblib"):
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # 目標層級：health_data/{date}/records/{time}
    root_ref = db.collection(root)

    # 要掃哪些日期文件
    if days is None:
        date_ids = [doc.id for doc in root_ref.stream()]
    else:
        today = datetime.now().date()
        date_ids = [(today - timedelta(days=i)).isoformat() for i in range(days)]

    processed = skipped = errors = 0
    for date_id in date_ids:
        recs = root_ref.document(date_id).collection("records")
        for doc in recs.stream():
            data = doc.to_dict() or {}

            # idempotent：如果已有 advice_updated_at 且沒開 force，就略過
            if data.get("advice_updated_at") and not force:
                skipped += 1
                continue

            feats, missing = to_features(data)
            if missing:  # 真的缺太多就跳過
                print(f"Skip {date_id}/{doc.id}: missing {missing}")
                skipped += 1
                continue

            try:
                state, tips = predict_and_advise(feats, model_path=model_path)
                advice_text = f"狀態：{state}。{tips[0]}"

                # 寫回同一筆文件（merge=true）
                doc.reference.set({
                    "state": state,
                    "advice_tips": tips,
                    "advice_text": advice_text,
                    "advice_updated_at": firestore.SERVER_TIMESTAMP,
                    # 同時補齊標準欄位，之後前端查詢更輕鬆
                    "temp_c": feats.get("temp_c"),
                    "condition": feats.get("condition"),
                    "is_raining": feats.get("is_raining"),
                }, merge=True)

                processed += 1
            except Exception as e:
                print(f"Error {date_id}/{doc.id}: {e}")
                errors += 1

    print(f"Done. processed={processed}, skipped={skipped}, errors={errors}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="path to service account JSON")
    ap.add_argument("--root", default="health_data", help="root collection name")
    ap.add_argument("--days", type=int, help="only process recent N days (default: all)")
    ap.add_argument("--force", action="store_true", help="recompute even if advice exists")
    ap.add_argument("--model", default="behavior_health_model.joblib", help="model path")
    args = ap.parse_args()
    run(args.key, root=args.root, days=args.days, force=args.force, model_path=args.model)

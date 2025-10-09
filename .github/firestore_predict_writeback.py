# -*- coding: utf-8 -*-
import re, sys, math, datetime as dt
from google.cloud import firestore

def parse_weather(x):
    """
    支援：
    - "Clouds, 24°C"
    - {"condition": "Clouds", "temp_c": 24}
    - "Clouds" / 24
    """
    cond, temp_c = None, None
    if isinstance(x, dict):
        cond = x.get("condition")
        temp_c = x.get("temp_c")
    elif isinstance(x, str):
        m = re.match(r"\s*([A-Za-z]+)\s*,\s*([\-]?\d+)\s*°?C\s*", x)
        if m:
            cond, temp_c = m.group(1), int(m.group(2))
        else:
            cond = x
    elif isinstance(x, (int, float)):
        temp_c = float(x)
    return cond, temp_c

def to_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def rule_based_state(feat):
    """
    即使心率等缺欄位，也能用「剩下的特徵」給出合理/保守的建議。
    你可以按需求調整門檻。
    """
    steps = feat.get("steps")
    avg_hr = feat.get("avg_heart_rate")
    max_hr = feat.get("max_heart_rate")
    temp_c = feat.get("temp_c")

    reasons = []

    # 以步數為最基本信號
    if steps is not None:
        if steps < 2000:
            state = "under_recovered"   # 太低，傾向恢復日
            reasons.append(f"very low steps ({steps})")
        elif steps > 12000:
            state = "overreached"       # 超高，注意恢復
            reasons.append(f"very high steps ({steps})")
        else:
            state = "ok"
            reasons.append(f"steps in range ({steps})")
    else:
        state = "ok"  # 沒步數就保守給 ok
        reasons.append("no steps -> default ok")

    # 心率若存在就調整置信
    if avg_hr is not None and avg_hr > 90:
        state = "under_recovered"
        reasons.append(f"high avg_hr ({avg_hr})")
    if max_hr is not None and max_hr > 170:
        state = "overreached"
        reasons.append(f"very high max_hr ({max_hr})")

    # 高溫下調整（如果有溫度）
    if temp_c is not None and temp_c >= 32 and state == "ok":
        state = "under_recovered"
        reasons.append(f"hot ({temp_c}C) -> recover")

    # 文案（可按你的產品調）
    tips = []
    if state == "under_recovered":
        tips += [
            "今天放慢節奏，做 20–30 分鐘輕鬆步行或伸展。",
            "補 300–500ml 水分，優先早睡 30 分鐘。",
        ]
    elif state == "overreached":
        tips += [
            "已經很努力！明天安排低強度日，避免連續高負荷。",
            "做 5–10 分鐘呼吸放鬆，補充電解質與蛋白質。",
        ]
    else:  # ok
        tips += [
            "保持現在的節奏，維持 30–45 分鐘中等強度活動。",
            "外出活動記得補水、防曬、防中暑。",
        ]
    return state, tips, reasons

def build_features(raw):
    """
    把各種 key 轉成統一特徵；缺欄位就不塞。
    也把 weather/溫度解析進來。
    """
    feat = {}
    # 支援多命名
    for k in ["steps"]:
        if k in raw: feat["steps"] = to_int(raw.get(k))

    # 心率：有就用
    for k in ["avgheartrate","avg_heart_rate","heartrate"]:
        if k in raw: feat["avg_heart_rate"] = to_float(raw.get(k)); break
    for k in ["maxheartrate","max_heart_rate"]:
        if k in raw: feat["max_heart_rate"] = to_float(raw.get(k)); break

    # weather
    cond, temp_c = None, None
    if "weather" in raw:
        cond, temp_c = parse_weather(raw.get("weather"))
    if "condition" in raw and "temp_c" in raw:
        # 另一種存法
        cond = cond or raw.get("condition")
        temp_c = temp_c if temp_c is not None else to_float(raw.get("temp_c"))
    if cond:   feat["condition"] = cond
    if temp_c is not None: feat["temp_c"] = temp_c

    return feat

def advise_one(doc_ref, raw, now_utc):
    """
    對單筆 record 給建議（即使資料不完整）。
    一律寫回 _advised，避免下輪重覆掃描。
    """
    feat = build_features(raw)
    missing = [k for k in ["steps","avg_heart_rate","max_heart_rate","temp_c"] if k not in feat]

    state, tips, reasons = rule_based_state(feat)

    # 組 advice map
    advice = {
        "state": state,
        "tips": tips,
        "features_used": feat,
        "missing_features": missing,
        "_advised": True,
        "_advised_at_utc": firestore.SERVER_TIMESTAMP,
        "_reason": reasons,
    }
    # 寫回 doc 欄位與子集合（版本 v1）
    doc_ref.set({"advice": advice, "_advised": True, "_advised_at_utc": now_utc}, merge=True)
    doc_ref.collection("advice").document("v1").set(advice, merge=True)
    return advice

def main(days=7, root="health_data"):
    db = firestore.Client()
    now = dt.datetime.utcnow()
    start_date = (now - dt.timedelta(days=days)).date().isoformat()
    end_date = now.date().isoformat()

    wrote = 0
    for date_doc in db.collection(root).list_documents():
        date_id = date_doc.id  # 2025-10-09
        if not (start_date <= date_id <= end_date):
            continue

        for rec_doc in date_doc.collection("records").list_documents():
            data = rec_doc.get().to_dict() or {}
            if data.get("_advised"):     # 已回覆過就略過
                print(f"SKIP(advised): {date_id}/{rec_doc.id}")
                continue

            try:
                adv = advise_one(rec_doc, data, now)
                wrote += 1
                print(f"WRITE: {date_id}/{rec_doc.id} -> {adv['state']}  miss={adv['missing_features']}  via={adv['_reason']}")
            except Exception as e:
                print(f"ERROR writing {date_id}/{rec_doc.id}: {e}", file=sys.stderr)

    print(f"[DONE] wrote={wrote}")

if __name__ == "__main__":
    # 這裡讀 CLI 參數略，保留你原本的參數解析。
    main()

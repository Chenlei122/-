# firestore_predict_writeback.py
import os, re, json, math, argparse, datetime as dt
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# 嘗試載入模型（可沒有）
MODEL = None
PREPROCESSOR = None
try:
    from joblib import load
    if os.path.exists("behavior_health_model.joblib"):
        MODEL_BUNDLE = load("behavior_health_model.joblib")
        # 你原本存的是整個 pipeline 就直接當成 MODEL 用
        MODEL = MODEL_BUNDLE
except Exception as e:
    print(f"[WARN] Load model failed: {e}")

# Firestore
import firebase_admin
from firebase_admin import credentials, firestore

THRESHOLDS = {
    "high_hr": 100,
    "very_high_hr": 120,
    "low_steps": 4000,
    "good_steps": 8000,
    "low_sleep": 6.0,
    "hot": 30.0,
    "cold": 15.0
}

def parse_weather(s: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    解析像 'Clouds, 29°C' -> ('Clouds', 29.0)
    """
    if not s or not isinstance(s, str): 
        return None, None
    # 先用逗號切
    parts = [p.strip() for p in s.split(",")]
    cond = parts[0] if parts else None
    temp_c = None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?C", s)
    if m:
        temp_c = float(m.group(1))
    return cond, temp_c

def infer_is_raining(condition: Optional[str]) -> int:
    if not condition: 
        return 0
    c = condition.lower()
    return 1 if any(k in c for k in ["rain", "drizzle", "thunderstorm"]) else 0

def estimate_activity_minutes(steps: Optional[int]) -> int:
    if steps is None: return 0
    # 粗估：每 100 步 ≈ 1 分鐘，限制在 0..120 之間
    return int(max(0, min(120, steps/100)))

def estimate_stress(avg_hr: Optional[int], steps: Optional[int]) -> int:
    """
    1~5 粗估壓力：心率越高、步數越高 → 壓力可能越高；步數低但心率高 → 恢復不佳
    """
    if avg_hr is None: avg_hr = 70
    if steps is None: steps = 0
    score = 1
    if avg_hr > THRESHOLDS["high_hr"]: score += 2
    if steps > THRESHOLDS["good_steps"]: score += 1
    if steps < THRESHOLDS["low_steps"] and avg_hr > THRESHOLDS["high_hr"]: score += 1
    return int(max(1, min(5, score)))

def rule_based_label(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    tips = []
    hr = row.get("avg_heart_rate", 70) or 70
    steps = row.get("steps", 0) or 0
    sleep = row.get("sleep_hours", 7.0) or 7.0
    temp = row.get("temp_c", None)
    cond = row.get("condition", "")

    # 環境建議
    if temp is not None and temp >= THRESHOLDS["hot"]:
        tips.append("天氣炎熱，請補充水分並降低運動強度。")
    if temp is not None and temp <= THRESHOLDS["cold"]:
        tips.append("天氣偏冷，熱身要充分，注意保暖。")
    if infer_is_raining(cond):
        tips.append("外出路面濕滑，務必注意防滑與補光。")

    # 行為建議（步數與心率）
    if steps < THRESHOLDS["low_steps"]:
        tips.append("今天活動量較少，建議分段加入 2–3 次 10 分鐘的輕快步行。")
    elif steps < THRESHOLDS["good_steps"]:
        tips.append("今天活動量還行，再走 10–15 分鐘會更好。")
    else:
        tips.append("今日活動量很棒，記得充分放鬆與伸展。")

    # 休息／恢復
    if sleep < THRESHOLDS["low_sleep"]:
        tips.append("最近睡眠偏少，今晚目標至少 7 小時，睡前 1 小時減少藍光與咖啡因。")

    # 狀態分類
    if hr > THRESHOLDS["very_high_hr"] or (hr > THRESHOLDS["high_hr"] and steps > THRESHOLDS["good_steps"]):
        state = "overreached"
        tips.append("心率偏高且活動量大，明日建議降強度，以伸展與低強度恢復為主。")
    elif (steps < THRESHOLDS["low_steps"] and hr > THRESHOLDS["high_hr"]) or (sleep < THRESHOLDS["low_sleep"] and hr > THRESHOLDS["high_hr"]):
        state = "under_recovered"
        tips.append("體能恢復可能不足，建議補眠、低強度活動及補水。")
    else:
        state = "ok"
        tips.append("狀態穩定，維持規律活動與補水即可。")

    # 去重
    tips = list(dict.fromkeys(tips))
    return state, tips

def to_model_features(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 Firestore 原始欄位 → 模型需要的特徵
    """
    condition, temp_c = parse_weather(doc.get("weather"))
    avg_hr = doc.get("avgheartrate") or doc.get("avg_heart_rate")
    max_hr = doc.get("maxheartrate") or doc.get("max_heart_rate")
    steps = doc.get("steps")

    feat = {
        "temp_c": temp_c if temp_c is not None else 24.0,
        "condition": condition or "Clear",
        "steps": int(steps) if steps is not None else 0,
        "avg_heart_rate": int(avg_hr) if avg_hr is not None else 70,
        "max_heart_rate": int(max_hr) if max_hr is not None else 120,
        "sleep_hours": float(doc.get("sleep_hours") or 7.0),
        "activity_minutes": int(doc.get("activity_minutes") or estimate_activity_minutes(steps)),
        "stress_level": int(doc.get("stress_level") or estimate_stress(avg_hr, steps)),
        "is_raining": int(doc.get("is_raining") if doc.get("is_raining") is not None else infer_is_raining(condition)),
        "commuting_mode": doc.get("commuting_mode") or "walk",
        # 可以把位置也帶回寫，但不是模型特徵
        "latitude": float(doc.get("latitude")) if doc.get("latitude") is not None else None,
        "longitude": float(doc.get("longitude")) if doc.get("longitude") is not None else None,
    }
    return feat

def model_or_rules(feat: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    先嘗試模型；若沒有模型，或預測不合理，退回規則式。
    """
    if MODEL is not None:
        try:
            df = pd.DataFrame([{
                "temp_c": feat["temp_c"],
                "condition": feat["condition"],
                "steps": feat["steps"],
                "avg_heart_rate": feat["avg_heart_rate"],
                "max_heart_rate": feat["max_heart_rate"],
                "sleep_hours": feat["sleep_hours"],
                "stress_level": feat["stress_level"],
                "activity_minutes": feat["activity_minutes"],
                "is_raining": feat["is_raining"],
                "commuting_mode": feat["commuting_mode"],
            }])
            pred = MODEL.predict(df)[0]
            # 你原本模型的 label 名稱：ok / under_recovered / overreached
            # 提示仍用規則式生成，或你也可維持原本模型 tips
            _, tips = rule_based_label(feat)
            return str(pred), tips
        except Exception as e:
            print(f"[WARN] model inference failed: {e}")
    # fallback
    return rule_based_label(feat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="service account json")
    ap.add_argument("--root", default="health_data", help="root collection")
    ap.add_argument("--hours", type=int, default=6, help="lookback hours")
    args = ap.parse_args()

    if not firebase_admin._apps:
        cred = credentials.Certificate(args.key)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # 時間窗
    now = dt.datetime.utcnow()
    since = now - dt.timedelta(hours=args.hours)

    root = db.collection(args.root)
    # 列出所有日期文件
    dates = sorted([d.id for d in root.list_documents()])
    processed = 0; wrote = 0

    for date_id in dates[-7:]:  # 最近 7 天就好
        recs = root.document(date_id).collection("records").list_documents()
        for r in recs:
            doc = r.get()
            data = doc.to_dict() or {}
            # 已經寫過 advice 就略過（你也可改成覆寫）
            if "advice" in data or data.get("_advised") == True:
                continue

            # Firestore 沒有標準時間欄位；用子文件 id（HH:MM:SS）+ 日期推時間
            # 若你的 Android 端有真正的 timestamp 欄位可直接用它：
            try:
                hh, mm, ss = map(int, doc.id.split(":"))
                naive = dt.datetime.strptime(date_id, "%Y-%m-%d").replace(hour=hh, minute=mm, second=ss)
            except Exception:
                naive = now  # fallback
            # UTC 假設：加減由你調整
            ts = naive
            if ts < since:
                continue

            feat = to_model_features(data)
            state, tips = model_or_rules(feat)

            advice_payload = {
                "state": state,
                "tips": tips,
                "features_used": {k: feat[k] for k in ["temp_c","condition","steps","avg_heart_rate","max_heart_rate","sleep_hours","stress_level","activity_minutes","is_raining","commuting_mode"]},
                "_advised": True,
                "_advised_at_utc": dt.datetime.utcnow()
            }

            # 你可以選擇回寫為子集合 / 欄位。這裡兩者都放：欄位方便看，子集合可擴充版本
            r.set({"advice": advice_payload, "_advised": True}, merge=True)
            r.collection("advice").document("v1").set(advice_payload)
            wrote += 1
            processed += 1

    print(f"[DONE] processed={processed}, wrote={wrote}")

if __name__ == "__main__":
    main()

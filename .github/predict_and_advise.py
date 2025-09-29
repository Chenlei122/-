
import pandas as pd
from joblib import load

ADVICE_MAP = {
    "under_recovered": [
        "今天先放輕量運動或休息，留意恢復。",
        "睡眠目標 7–9 小時，睡前 1 小時降低螢幕光。",
        "補充蛋白與溫熱飲品；若天氣冷(<20°C)請加外套。",
        "如果在下雨，外出記得帶雨具，儘量室內活動。"
    ],
    "ok": [
        "狀態穩定，維持中等強度活動 30–60 分鐘。",
        "每 2–3 小時補水一次；工作間站起來活動 3–5 分鐘。"
    ],
    "overreached": [
        "有過度訓練跡象，今天把強度降到 30–50%，避免間歇高強度。",
        "專注伸展與走路，若心率居高或不適，請監測並視情況就醫。",
        "炎熱天氣(>30°C)務必增加電解質補充。"
    ]
}

def enrich_weather_tips(row, tips):
    t = row.get("temp_c", 24)
    cond = str(row.get("condition","")).lower()
    if "rain" in cond or "drizzle" in cond or row.get("is_raining",0)==1:
        tips = ["可能降雨，帶雨具並留意濕滑路面。"] + tips
    if t < 20:
        tips.append("氣溫偏低，注意保暖，熱身時間加長。")
    if t > 30:
        tips.append("高溫警示，避免正午外出並加強補水。")
    return tips

def predict_and_advise(record: dict, model_path="behavior_health_model.joblib"):
    model = load(model_path)
    df = pd.DataFrame([record])
    pred = model.predict(df)[0]
    tips = ADVICE_MAP.get(pred, ["保持日常作息與規律活動。"])
    tips = enrich_weather_tips(record, tips)
    return pred, tips

if __name__ == "__main__":
    sample = {
        "latitude": 22.99, "longitude": 120.25, "temp_c": 24.0, "condition": "Clouds",
        "steps": 6500, "avg_heart_rate": 74, "max_heart_rate": 160,
        "sleep_hours": 6.5, "stress_level": 3, "activity_minutes": 40,
        "is_raining": 0, "commuting_mode": "scooter"
    }
    pred, tips = predict_and_advise(sample)
    print("Predicted state:", pred)
    print("Advice:")
    for t in tips:
        print("-", t)

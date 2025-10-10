# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import re

# 這兩個庫若缺失，會自動走 fallback，不讓整條管線爆掉
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None          # noqa: N816
try:
    from joblib import load  # type: ignore
except Exception:  # pragma: no cover
    load = None

__all__ = ["make_advice", "extract_features"]  # 明確導出給 workflow 匯入

# ----------------------------
# 1) 欄位別名與基本解析工具
# ----------------------------
ALIASES: Dict[str, List[str]] = {
    "avg_hr":   ["avgheartrate", "avg_heart_rate", "average_heart_rate", "heartrate", "heart_rate", "avgHR"],
    "steps":    ["steps", "step", "stepcount", "daily_steps"],
    "weather":  ["weather", "condition", "conditions"],
    "latitude": ["lat", "latitude"],
    "longitude":["lng", "lon", "longitude"],
    "min_hr":   ["minheartrate", "min_heart_rate"],
    "max_hr":   ["maxheartrate", "max_heart_rate"],
}

NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")
TEMP_C = re.compile(r"([-+]?\d+)\s*°?C", re.IGNORECASE)

def _first_present(d: Dict[str, Any], keys: List[str]) -> tuple[Any | None, str | None]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k], k
    return None, None

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = NUMERIC_LIKE.search(s)
    return float(m.group()) if m else None

def _clean_weather(s: Any) -> Dict[str, Any]:
    """支援 dict/str/None；回傳 {label, temp_c, raw} 的小字典。"""
    if s is None:
        return {}
    if isinstance(s, dict):
        label = str(s.get("label") or s.get("condition") or "").strip()
        temp = _to_float(s.get("temp_c"))
        out = {"raw": s}
        if label: out["label"] = label
        if temp is not None: out["temp_c"] = temp
        return out

    s_str = str(s)
    out: Dict[str, Any] = {"raw": s_str}
    m = TEMP_C.search(s_str)
    if m:
        try:
            out["temp_c"] = float(m.group(1))
        except Exception:
            pass
    out["label"] = s_str.split(",")[0].strip()
    return out

def extract_features(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    回傳 (features, meta)
    features: 後續建議要用到的標準化特徵
    meta:     用了哪些別名、缺失欄位、原始 keys 等
    """
    features: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"aliases_used": {}, "missing_features": []}

    for canon, alias_list in ALIASES.items():
        val, key_used = _first_present(record, alias_list)
        if canon in ("avg_hr", "min_hr", "max_hr", "steps", "latitude", "longitude"):
            val = _to_float(val)
        elif canon == "weather":
            val = _clean_weather(val)

        if val is not None and (canon != "weather" or val):  # weather dict 不為空才算
            features[canon] = val
            meta["aliases_used"][canon] = key_used
        else:
            meta["missing_features"].append(canon)

    meta["raw_keys"] = sorted(list(record.keys()))
    meta["features_used"] = sorted(list(features.keys()))
    return features, meta

# ----------------------------
# 2) 模型＋規則：雙路徑
# ----------------------------
ADVICE_MAP: Dict[str, List[str]] = {
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
    ],
}

def _enrich_weather_tips(weather: Dict[str, Any] | None, tips: List[str]) -> List[str]:
    if not weather:
        return tips
    t = weather.get("temp_c")
    label = str(weather.get("label", "")).lower()
    if any(k in label for k in ("rain", "drizzle", "shower", "storm")):
        tips = ["可能降雨，帶雨具並留意濕滑路面。"] + tips
    if isinstance(t, (int, float)):
        if t < 20:
            tips.append("氣溫偏低，注意保暖，熱身時間加長。")
        if t > 30:
            tips.append("高溫警示，避免正午外出並加強補水。")
    return tips

def _fallback_rules(feat: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """沒有模型或模型失敗時的一套保守規則。"""
    tips: List[str] = []
    state = "ok"
    confidence = 0.5

    if "avg_hr" in feat and feat["avg_hr"] is not None:
        hr = float(feat["avg_hr"])
        confidence += 0.2
        if hr > 110:
            tips.append("平均心率偏高，今天運動後請拉長冷身、補充電解質並留意休息。")
            state = "watch"
        elif hr < 45:
            tips.append("平均心率偏低，如非運動員體質，建議留意疲勞程度與睡眠。")

    if "max_hr" in feat and feat.get("avg_hr") is not None:
        try:
            if float(feat["max_hr"]) > float(feat["avg_hr"]) + 40:
                tips.append("最高心率尖峰較高，建議暖身更充分並避免突然爆發。")
        except Exception:
            pass

    if "steps" in feat and feat["steps"] is not None:
        confidence += 0.15
        s = int(feat["steps"])
        if s < 3000:
            tips.append("今日步數偏低，安排 10–15 分鐘散步會有幫助。")
        elif s < 8000:
            tips.append("步數中等，若精神狀況允許可再加 15 分鐘走路。")
        else:
            tips.append("步數已達標，記得拉筋與補水。")

    tips = _enrich_weather_tips(feat.get("weather"), tips)

    if not tips:
        tips = ["資料有限，維持規律作息、補充水分與輕量活動。"]
        confidence = 0.3

    confidence = max(0.0, min(1.0, confidence))
    return {
        "advice": {
            "state": state,
            "tips": tips,
            "features_used": meta["features_used"],
            "missing_features": meta["missing_features"],
            "model_version": "fallback-v1",
            "confidence": confidence,
        }
    }

def _model_predict(feat: Dict[str, Any], model_path: str) -> tuple[Optional[str], List[str]]:
    """
    使用 joblib 模型；若失敗或模型不存在，回傳 (None, [])。
    期望模型吃到的欄位：依你訓練時的 schema 調整。
    """
    p = Path(model_path)
    if (load is None) or (pd is None) or (not p.exists()):
        return None, []

    try:
        model = load(p)  # type: ignore[arg-type]
        row = {
            "latitude":  feat.get("latitude"),
            "longitude": feat.get("longitude"),
            "temp_c":    (feat.get("weather") or {}).get("temp_c"),
            "condition": (feat.get("weather") or {}).get("label"),
            "steps":     feat.get("steps"),
            "avg_heart_rate": feat.get("avg_hr"),
            "max_heart_rate": feat.get("max_hr"),
            "min_heart_rate": feat.get("min_hr"),
        }
        df = pd.DataFrame([row])  # type: ignore[arg-type]
        pred = model.predict(df)[0]
        base_tips = ADVICE_MAP.get(str(pred), ["保持日常作息與規律活動。"])
        return str(pred), base_tips
    except Exception:
        return None, []

# ----------------------------
# 3) 對外接口：回傳 Firestore 可直接 merge 的 payload
# ----------------------------
def make_advice(record: Dict[str, Any], model_path: str = "behavior_health_model.joblib") -> Dict[str, Any]:
    """
    主函式：先做特徵抽取；有模型就跑模型＋補天氣提示；失敗則走 fallback。
    回傳的 dict 可直接 Firestore set(..., merge=True)。
    """
    feat, meta = extract_features(record)

    pred, model_tips = _model_predict(feat, model_path)
    if pred is not None:
        tips = _enrich_weather_tips(feat.get("weather"), list(model_tips))
        payload = {
            "advice": {
                "state": str(pred),
                "tips": tips,
                "features_used": meta["features_used"],
                "missing_features": meta["missing_features"],
                "model_version": "model-v1",
                "confidence": 0.7 if meta["missing_features"] else 0.85,
            }
        }
    else:
        payload = _fallback_rules(feat, meta)

    if "weather" in feat:
        payload["weather"] = feat["weather"]
    if "latitude" in feat and "longitude" in feat:
        payload["location"] = {"lat": feat["latitude"], "lng": feat["longitude"]}

    payload["_advised"] = True
    return payload

# ----------------------------
# 4) 本地快速測試
# ----------------------------
if __name__ == "__main__":
    sample = {
        "latitude": 22.99, "longitude": 120.25, "weather": "Clouds, 24°C",
        "steps": 6500, "avg_heart_rate": 74, "max_heart_rate": 160,
        "sleep_hours": 6.5, "stress_level": 3, "activity_minutes": 40,
        "is_raining": 0, "commuting_mode": "scooter"
    }
    out = make_advice(sample)
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2))

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import re
import random  # 用來隨機挑選溫柔的話

# ---------------------------------------------------------
# 依賴庫檢查 (Fallback 機制)
# ---------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None          # noqa: N816
try:
    from joblib import load  # type: ignore
except Exception:  # pragma: no cover
    load = None

__all__ = ["make_advice", "extract_features"]

# ==========================================
# 🚀 效能優化核心：模型快取變數
# ==========================================
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None

def _get_model(model_path: str):
    """
    快取機制：如果模型已經載入過且路徑一樣，直接回傳記憶體裡的那份。
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    
    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == model_path:
        return _CACHED_MODEL
    
    p = Path(model_path)
    if (load is None) or (pd is None) or (not p.exists()):
        return None
        
    try:
        # print(f"[Model] Loading model from disk: {model_path} (Run once)")
        _CACHED_MODEL = load(p)
        _CACHED_MODEL_PATH = model_path
        return _CACHED_MODEL
    except Exception as e:
        print(f"[Model] Load failed: {e}")
        return None

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
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    m = NUMERIC_LIKE.search(s)
    return float(m.group()) if m else None

def _clean_weather(s: Any) -> Dict[str, Any]:
    if s is None: return {}
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
    features: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"aliases_used": {}, "missing_features": []}

    for canon, alias_list in ALIASES.items():
        val, key_used = _first_present(record, alias_list)
        if canon in ("avg_hr", "min_hr", "max_hr", "steps", "latitude", "longitude"):
            val = _to_float(val)
        elif canon == "weather":
            val = _clean_weather(val)

        if val is not None and (canon != "weather" or val):
            features[canon] = val
            meta["aliases_used"][canon] = key_used
        else:
            meta["missing_features"].append(canon)

    meta["raw_keys"] = sorted(list(record.keys()))
    meta["features_used"] = sorted(list(features.keys()))
    return features, meta

# ----------------------------
# 2) 🌿 溫柔多變的對話庫 (Gentle & Varied Advice)
# ----------------------------
# 這裡只有一種「溫柔」風格，但準備了很多種不同的說法 (Variations)

ADVICE_POOLS: Dict[str, List[List[str]]] = {
    "under_recovered": [
        # Variation 1
        [
            "親愛的，今天身體似乎有點累了呢 🌿",
            "給自己一點空間，今晚早點休息好嗎？💤",
            "如果覺得冷，記得多加件衣服，別著涼囉。🧥"
        ],
        # Variation 2
        [
            "今天稍微放慢腳步吧，不需要每天都衝刺的 🐢",
            "把「好好睡覺」當作今天的任務，睡前少看手機喔 📵",
            "多喝點溫熱的水，讓身體暖活起來 🍵"
        ],
        # Variation 3
        [
            "狀態顯示電量偏低 🪫 沒關係，這是身體在提醒你該充電了。",
            "今天不適合太激烈的運動，簡單伸展一下就好 🧘‍♀️",
            "好好照顧自己，休息是為了走更長遠的路 ✨"
        ],
        # Variation 4
        [
            "是不是最近比較忙碌呢？身體想休息了 🛌",
            "今晚給自己一個熱水澡，放鬆一下緊繃的神經 🛁",
            "別給自己壓力，今天不做運動也完全沒問題的！👌"
        ]
    ],
    "ok": [
        # Variation 1
        [
            "今天的狀態看起來很不錯喔！✨",
            "維持這樣舒服的節奏就好，不用刻意改變什麼。",
            "如果天氣好，出去走走曬曬太陽會很舒服的 ☀️"
        ],
        # Variation 2
        [
            "一切都很平穩，是你把自己照顧得很好的證明 👍",
            "適合做點中等強度的活動，流流汗心情會變好喔！🏃‍♀️",
            "記得工作空檔站起來動一動，喝口水 💧"
        ],
        # Variation 3
        [
            "Good vibes! 🌟 身體感覺輕盈又穩定。",
            "今天可以試著多走一點路，或是做個 30 分鐘有氧 🔥",
            "保持這份好心情，享受今天吧！😊"
        ],
        # Variation 4
        [
            "數據顯示今天狀態很平衡 ⚖️ 很棒喔！",
            "按照你原本的計畫進行就好，你做得很好。",
            "別忘了多補充水分，讓代謝保持順暢 🥤"
        ]
    ],
    "overreached": [
        # Variation 1
        [
            "嘿，你最近真的太努力了，身體有點抗議囉 🛑",
            "今天請對自己「好一點」，運動強度減半，或者乾脆休息。",
            "留意一下心跳是不是比較快？那是身體在撒嬌討拍了 💓"
        ],
        # Variation 2
        [
            "小提醒 🔔 你的負荷量有點偏高，小心別累壞了。",
            "今天先不要做高強度的訓練，散散步就好 🚶‍♀️",
            "多吃點營養的東西，幫助身體修復 🥗"
        ],
        # Variation 3
        [
            "Whoa~ 暫停一下 ✋ 讓我們把腳步放慢。",
            "過度努力會受傷的，今天專注在放鬆和伸展吧 🧘‍♂️",
            "今晚試著比平常早一小時睡覺，好嗎？🛌"
        ],
         # Variation 4
        [
            "親愛的，別給自己太大的壓力 ❤️ 身體需要緩衝。",
            "今天適合「耍廢」一下，讓肌肉徹底放鬆。",
            "如果覺得疲勞感很重，那是身體在向你求救喔，請多休息。🌿"
        ]
    ]
}

def _get_random_advice(state: str) -> List[str]:
    """從溫柔對話庫中隨機撈一組建議"""
    pool = ADVICE_POOLS.get(state)
    if not pool:
        # 預設的溫柔備案
        return ["今天看起來一切正常，保持開心的心情喔！😊"]
    return random.choice(pool)

def _enrich_weather_tips(weather: Dict[str, Any] | None, tips: List[str]) -> List[str]:
    """根據天氣加上貼心的提醒"""
    if not weather:
        return tips
    
    t = weather.get("temp_c")
    label = str(weather.get("label", "")).lower()
    
    # 降雨判斷 - 溫柔提醒
    if any(k in label for k in ("rain", "drizzle", "shower", "storm")):
        tips.insert(0, "外面滴滴答答在下雨 ☔ 出門要帶傘，小心腳下濕滑喔。")
    elif "cloud" in label:
        tips.insert(0, "今天是個舒服的陰天 ☁️ 很適合出去透透氣。")
    elif "clear" in label or "sun" in label:
        tips.insert(0, "陽光很棒呢！☀️ 看見陽光心情也會變好喔。")

    # 溫度判斷 - 溫柔關心
    if isinstance(t, (int, float)):
        if t < 18:
            tips.append("氣溫有點低 🥶 記得圍個圍巾或多穿件外套，別感冒了。🧣")
        elif t > 30:
            tips.append("天氣好像有點熱 🥵 在外面要記得多喝水，找陰涼處休息喔。💧")
    
    return tips

def _fallback_rules(feat: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """沒有模型時的備用規則 (一樣走溫柔路線)"""
    tips: List[str] = []
    state = "ok"
    confidence = 0.5

    if "avg_hr" in feat and feat["avg_hr"] is not None:
        hr = float(feat["avg_hr"])
        confidence += 0.2
        if hr > 110:
            tips.append("心跳好像有點快 💓 試著深呼吸，讓自己平靜下來。")
            state = "watch"
        elif hr < 45:
            tips.append("心跳比較慢一點 📉 最近是不是太累了？要多休息喔。")

    if "steps" in feat and feat["steps"] is not None:
        confidence += 0.15
        s = int(feat["steps"])
        if s < 3000:
            tips.append("今天動得稍微少了一點點 🤏 待會起來走個 10 分鐘好嗎？")
        elif s < 8000:
            tips.append("步數還不錯喔 👍 再散個步就更完美了！")
        else:
            tips.append("哇，今天走很多路呢！🌟 辛苦了，回家記得抬腿放鬆一下。")

    tips = _enrich_weather_tips(feat.get("weather"), tips)

    if not tips:
        tips = ["資料稍微有點少 🤔 不過不管怎樣，都要記得多喝水、早點睡喔！"]
        confidence = 0.3

    confidence = max(0.0, min(1.0, confidence))
    return {
        "advice": {
            "state": state,
            "tips": tips,
            "features_used": meta["features_used"],
            "missing_features": meta["missing_features"],
            "model_version": "fallback-v1-gentle",
            "confidence": confidence,
        }
    }

def _model_predict(feat: Dict[str, Any], model_path: str) -> tuple[Optional[str], List[str]]:
    """使用快取模型進行預測"""
    model = _get_model(model_path)
    if model is None:
        return None, []

    try:
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
        
        # 使用隨機挑選的「溫柔」建議
        base_tips = _get_random_advice(str(pred))
        return str(pred), base_tips
    except Exception as e:
        return None, []

# ----------------------------
# 3) 對外接口
# ----------------------------
def make_advice(record: Dict[str, Any], model_path: str = "behavior_health_model.joblib") -> Dict[str, Any]:
    """
    主函式：回傳的 dict 可直接 Firestore set(..., merge=True)。
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
                "model_version": "model-v1-gentle",
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
# 4) 本地快速測試 (測試隨機溫柔語氣)
# ----------------------------
if __name__ == "__main__":
    sample = {
        "latitude": 22.99, "longitude": 120.25, 
        "weather": "Rain, 16°C",  # 測試下雨 + 低溫
        "steps": 2500, 
        "avg_heart_rate": 65, 
        "max_heart_rate": 120,
    }
    
    print("--- 第一次測試 (看看她說什麼) ---")
    out1 = make_advice(sample, "dummy_path") 
    import json
    print(json.dumps(out1["advice"]["tips"], ensure_ascii=False, indent=2))
    
    print("\n--- 第二次測試 (看看她會不會換句話說) ---")
    out2 = make_advice(sample, "dummy_path")
    print(json.dumps(out2["advice"]["tips"], ensure_ascii=False, indent=2))
# -*- coding: utf-8 -*-
import argparse
import datetime
import re
import sys
from typing import Dict, Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, RetryError

# ---------------------------------------------------------
# 1. 外部 AI 載入
# ---------------------------------------------------------
_HAVE_MAKE_ADVICE = False
try:
    from predict_and_advise import make_advice as _external_make_advice
    _HAVE_MAKE_ADVICE = True
except ImportError:
    print("⚠️ 警告: 找不到 predict_and_advise.py，將使用 V12 智慧規則。")

# ---------------------------------------------------------
# 2. 環境感知輔助工具
# ---------------------------------------------------------
_NUMERIC_LIKE = re.compile(r"[-+]?\d*\.?\d+")

def to_float(x, d: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        m = _NUMERIC_LIKE.search(str(x))
        return float(m.group()) if m else d
    except Exception: return d

def parse_weather_temp(weather_str: str) -> float:
    """從 'Clouds, 18°C' 提取溫度數字"""
    try:
        if not weather_str: return 25.0
        match = re.search(r"(\d+)°C", weather_str)
        return float(match.group(1)) if match else 25.0
    except: return 25.0

def get_hour_from_timestamp(ts) -> int:
    """解析 13 位毫秒時間戳記獲取小時"""
    try:
        dt = datetime.datetime.fromtimestamp(int(ts) / 1000.0)
        return dt.hour
    except: return 12

def rule_based_fallback(data: Dict[str, Any]) -> Dict[str, Any]:
    """智慧化建議邏輯：整合時間與溫度"""
    steps = to_float(data.get("steps"), 0)
    weather_raw = data.get("weather", "")
    temp = parse_weather_temp(weather_raw)
    hour = get_hour_from_timestamp(data.get("timestamp", 0))
    
    tips = []
    # 定義環境閾值
    is_late_night = hour >= 21 or hour <= 5
    is_cold = temp < 16
    
    if steps < 2000:
        if is_late_night:
            tips.append(f"現在是深夜 {hour} 點，活動量偏低，建議在家簡單伸展就好，早點休息 🌙")
        elif is_cold:
            tips.append(f"外面氣溫僅 {temp}°C 較冷，建議在室內原地踏步或居家運動來增加活動量 🏠")
        else:
            tips.append(f"現在氣溫 {temp}°C 很舒適，適合到戶外走走達成步數目標 🚶")
    else:
        if is_late_night:
            tips.append("今日步數已達標！深夜時分，請放鬆心情準備入睡 ✨")
        else:
            tips.append("太棒了！你的活動量非常充足，請繼續保持 🌟")

    return {
        "advice": {
            "state": "ok", 
            "tips": tips, 
            "model_version": "v12-context-aware",
            "context": {"hour": hour, "temp": temp} 
        },
        "_advised": True
    }

def generate_payload(data: Dict[str, Any], model_path: str, user_id: str) -> Dict[str, Any]:
    if _HAVE_MAKE_ADVICE:
        try:
            result = _external_make_advice(data, model_path=model_path)
        except:
            result = rule_based_fallback(data)
    else:
        result = rule_based_fallback(data)
    
    result["user_id"] = user_id
    result["created_at"] = firestore.SERVER_TIMESTAMP
    return result

# ---------------------------------------------------------
# 3. Firestore 核心邏輯
# ---------------------------------------------------------
def init_firestore(key_path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def batch_write_safe(batch):
    try:
        batch.commit()
        return True
    except Exception as e:
        print(f"❌ 批次寫入失敗: {e}")
        return False

def process_user_date(db, user_id, date_str, model_path, args) -> tuple:
    # 來源：/users/{id}/health_data/{date}/records
    source_ref = db.collection("users").document(user_id)\
                   .collection("health_data").document(date_str)\
                   .collection("records")
    
    # 目標：/users/{id}/advice_results
    target_ref = db.collection("users").document(user_id).collection("advice_results")

    try:
        query = source_ref.limit(args.limit)
        docs = list(query.stream(timeout=10))
    except (ResourceExhausted, RetryError):
        print(f"\n❌ [Quota Exceeded] 額度已耗盡！")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 讀取發生未知錯誤: {e}")
        return 0, 0, 0

    if not docs: return 0, 0, 0

    batch = db.batch()
    count = processed = skipped = 0

    for doc in docs:
        data = doc.to_dict()
        if (not args.force) and (data.get("_advised") is True):
            skipped += 1
            continue

        payload = generate_payload(data, model_path, user_id)
        if not payload: continue

        new_doc_id = f"{user_id}_{date_str}_{doc.id}"
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
# 4. 主程式
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--target-user", type=str, default="j3tgphl0e9gj2hPRaT6aAOhH1922")
    parser.add_argument("--model", default="behavior_health_model.joblib")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    db = init_firestore(args.key)
    
    print("="*60)
    print(f"🚀 AI 建議生成器 (V12: 環境感知增量版)")
    print("="*60)
    print(f"👤 目標使用者: {args.target_user}")
    print(f"📂 輸出集合:   advice_results")
    print("-" * 60)

    today = datetime.date.today()
    target_dates = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days + 1)]

    for date_str in target_dates:
        print(f"🔍 [檢查] 日期: {date_str}")
        total, skipped, processed = process_user_date(db, args.target_user, date_str, args.model, args)
        if total > 0:
            print(f"   🎉 掃描 {total} 筆 -> 新增 {processed} 筆 (略過 {skipped})")
        else:
            print(f"   💤 無資料")

    print("-" * 60)
    print(f"✅ 全部完成！")

if __name__ == "__main__":
    main()
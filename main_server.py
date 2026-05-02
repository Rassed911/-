from fastapi import FastAPI
from catboost import CatBoostClassifier, Pool
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# 1. Описываем входные данные
class CustomerData(BaseModel):
    Income: float
    Age: int
    Total_Spending: float
    Total_Purchases: int
    Children: int
    Recency: int

# 2. Инициализация приложения
app = FastAPI(title="Marketing Prediction API")

# 3. Загружаем модель и фичи при старте
# Убедись, что эти файлы лежат в той же папке!
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')
features = joblib.load('model_features.joblib')

@app.get("/")
def home():
    return {"message": "API работает!"}

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        data = pd.DataFrame([customer.dict()])
        
        # 1. Предсказание (делаем максимально надежно)
        prediction = model.predict(data)
        # Извлекаем число, даже если это массив [1] или [[1]]
        pred_value = int(np.array(prediction).flatten()[0])
        
        # 2. Вероятность
        proba_array = model.predict_proba(data)
        probability = float(np.array(proba_array).flatten()[pred_value])
        
        # 3. Важность признаков
        data_pool = Pool(data)
        importances = model.get_feature_importance(data=data_pool)
        
        # Очищаем от NaN и выпрямляем в обычный список
        clean_importances = np.nan_to_num(importances, nan=0.0).flatten()
        importance_map = {feat: float(val) for feat, val in zip(features, clean_importances)}
        
        return {
            "prediction": pred_value,
            "status": "Купит" if pred_value == 1 else "Не купит",
            "confidence": round(probability, 2),
            "importances": importance_map
        }
    except Exception as e:
        print(f"Backend Error: {str(e)}")
        return {"error": str(e)}
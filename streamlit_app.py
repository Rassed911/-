import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Marketing AI", layout="centered")

st.title("🚀 Прогноз маркетинговой кампании")
st.write("Введите данные клиента, чтобы узнать вероятность покупки.")

# 1. Форма ввода данных (интерфейс)
with st.sidebar:
    st.header("Параметры клиента")
    income = st.number_input("Годовой доход ($)", min_value=0, value=50000)
    age = st.slider("Возраст", 18, 90, 35)
    spending = st.number_input("Общие траты", min_value=0, value=500)
    purchases = st.number_input("Количество покупок", min_value=0, value=10)
    children = st.selectbox("Количество детей", [0, 1, 2, 3])
    recency = st.slider("Дней с последней покупки", 0, 100, 10)

# 2. Логика запроса к FastAPI
if st.button("Предсказать отклик"):
    # Формируем данные в формате JSON, который ждет наш FastAPI
    payload = {
        "Income": income,
        "Age": age,
        "Total_Spending": spending,
        "Total_Purchases": purchases,
        "Children": children,
        "Recency": recency
    }

    try:
        # Отправляем POST запрос в наш Docker-контейнер (или локальный сервер)
        response = requests.post("http://marketing_api_service:8000/predict", json=payload)
        result = response.json()
        

        # 3. Красивый вывод результата
        if result["prediction"] == 1:
            st.success(f"### ✅ Результат: {result['status']}")
        else:
            st.error(f"### ❌ Результат: {result['status']}")
            
        st.metric("Уверенность модели", f"{result['confidence']*100}%")
        
    
        st.divider() # Красивая линия-разделитель
        st.subheader("💡 Почему модель так решила?")
        
        # Превращаем пришедший словарь с весами в таблицу для графика
        imp_data = pd.DataFrame({
            "Признак": list(result["importances"].keys()),
            "Влияние": list(result["importances"].values())
        }).sort_values(by="Влияние", ascending=True)

        # Рисуем горизонтальный график
        st.bar_chart(data=imp_data, x="Признак", y="Влияние", horizontal=True)
        
        st.info("Чем длиннее полоска, тем сильнее этот фактор повлиял на итоговый прогноз.")
    except Exception as e:
        st.error(f"Ошибка подключения к API: {e}")
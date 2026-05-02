import pandas as pd
from catboost import CatBoostClassifier
import joblib
import sqlite3
# 1. Загрузка данных (используем твой очищенный датасет)
print("Подключаюсь к базе данных...")
# 1. Устанавливаем соединение с базой
conn = sqlite3.connect('marketing.db')
# 2. Пишем SQL-запрос (самое важное!)
query = "SELECT * FROM customers"
# 3. Загружаем данные в Pandas через SQL
df = pd.read_sql(query, conn)
conn.close()
print(f"Данные из SQL получены! Размер: {df.shape}")
# 2. Создание новых признаков (те, что мы обсуждали)
df['Age'] = 2021 - df['Year_Birth']
mnt_cols = [col for col in df.columns if 'Mnt' in col]
df['Total_Spending'] = df[mnt_cols].sum(axis=1)
num_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[num_cols].sum(axis=1)
df['Children'] = df['Kidhome'] + df['Teenhome']
# 3. Выбор признаков для модели
# Мы берем только то, что реально влияет на покупку
features = ['Income', 'Age', 'Total_Spending', 'Total_Purchases', 'Children', 'Recency']
X = df[features]
y = df['Response']
# 4. Обучение CatBoost
# Мы добавим параметр auto_class_weights, чтобы модель лучше видела "редких" покупателей
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    auto_class_weights='Balanced',
    verbose=100 # будет показывать прогресс каждые 100 шагов
)
model.fit(X, y)
# 5. СОХРАНЕНИЕ (Самый важный этап для инженера)
# Сохраняем саму модель
model.save_model('catboost_model.cbm')

# Сохраняем список колонок, чтобы API знал, в каком порядке подавать данные
joblib.dump(features, 'model_features.joblib')

print("Успех! Модель и признаки сохранены.")
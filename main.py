import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#-------------Загрузка данных-----------------------
df = pd.read_csv(r'G:\Projects_Py\Analiz_ludei_marketing\marketing_campaign.csv', sep='\t')
#print(df.head(5))
#print(df.info())
#------------Очистка данных от мусора и аномалий-------------
df = df.dropna(subset=['Income']) #Удаляю нулевые значения из колонки(потеряю не так много данных)
df = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1) #Значения из них не нужны,тк там везде одинаковые значения = никакой инормативности
df = df[df['Year_Birth'] > 1940] #Удаляю слишком старых клиентов,это  больше похоже на аномалию,чем на полезную информацию
df = df[df['Income'] < 120000] #Удаляю 'олигархов' с нереалистичным доходом
#print(f'Размер таблицы:{df.shape}')
if 'Age' not in df.columns:
    df['Age'] = 2021 - df['Year_Birth'] #Возраст покупателя
if 'Total_Spending' not in df.columns:
    mnt_cols = [col for col in df.columns if 'Mnt' in col]
    df['Total_Spending'] = df[mnt_cols].sum(axis=1)#Общее количество трат на все категории продуктов
if 'Total_Purchases' not in df.columns:
    num_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[num_cols].sum(axis=1) #Общее количество покупок
df['Children'] = df['Kidhome'] + df['Teenhome'] #Общее количество детей

#-------------Визуализация данных--------------------
# Распределение возраста
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Распределение клиентов по возрасту')
plt.xlabel('Возраст')
plt.ylabel('Количество клиентов')
plt.show()
#Доходы vs траты
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Income', y='Total_Spending', alpha=0.5, color='coral')
plt.title('Связь дохода и общих трат')
plt.xlabel('Годовой доход')
plt.ylabel('Всего потрачено')
plt.show()
#Траты в зависимости от количества детей
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Children', y='Total_Spending', palette='Set2')
plt.title('Траты клиентов в зависимости от количества детей')
plt.xlabel('Количество детей в семье')
plt.ylabel('Общие траты')
plt.show()
#Матрица корреляций
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Матрица корреляции всех признаков')
plt.show()
#Так как на матрице показало что вино один из показателей высокого дохода, посмотрим кто больше всего покупает вино
#1) Возраст и вино
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Age', y='MntWines', color='darkred')
plt.title('Зависимость трат на вино от возраста')
plt.show()
#2) Семейное положение и вино
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Marital_Status', y='MntWines', palette='vlag')
plt.title('Траты на вино в зависимости от семейного положения')
plt.show()
#3) Дети и вино
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Children', y='MntWines', palette='Reds')
plt.title('Как количество детей влияет на покупку вина')
plt.show()

#----------------Работа с данными------------------
features = ['Income', 'Total_Spending', 'Age', 'Total_Purchases'] #Признаки для сегментации клиентов
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)#Масштабируем данные,чтобы не отвалилось нужное
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) #Обучаем модель для предсказания
df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df.groupby('Cluster')[features].mean())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

features_for_model = ['Income', 'Age', 'Total_Spending', 'Total_Purchases', 'Children', 
                     'Recency', 'NumWebVisitsMonth', 'Cluster']
X = df[features_for_model]
y = df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#На сколько хорошо модель предсказывает покупателей
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).iloc[:-1, :2].T # Берем только важные классы
plt.figure(figsize=(8, 4))
sns.heatmap(report_df, annot=True, cmap='RdYlGn', cbar=False)
plt.title('Насколько хорошо модель предсказывает покупателей?')
plt.show()

#Самые ваажные признаки для модели
importance = pd.DataFrame({'Feature': features_for_model, 'Importance': model.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature', palette='magma')
plt.title('Что больше всего влияет на решение клиента? (Важность признаков)')
plt.show()





#print(df.describe().T)

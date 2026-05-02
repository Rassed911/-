import sqlite3
import pandas as pd

# 1. Загружаем твой CSV файл
df = pd.read_csv(r'G:\Projects_Py\Analiz_ludei_marketing\marketing_campaign.csv', sep='\t')

# 2. Подключаемся к SQLite (если файла нет, он создастся сам)
conn = sqlite3.connect('marketing.db')

# 3. Записываем данные в таблицу 'customers'
# if_exists='replace' означает, что если таблица есть, мы её перезапишем
df.to_sql('customers', conn, if_exists='replace', index=False)

print("Данные успешно перенесены из CSV в базу данных SQLite (marketing.db)!")
conn.close()
import sqlite3
import pandas as pd

df = pd.read_csv(r'G:\Projects_Py\Analiz_ludei_marketing\marketing_campaign.csv', sep='\t')

conn = sqlite3.connect('marketing.db')

# if_exists='replace' означает, что если таблица есть, мы её перезапишем
df.to_sql('customers', conn, if_exists='replace', index=False)

print("Данные успешно перенесены из CSV в базу данных SQLite (marketing.db)!")
conn.close()

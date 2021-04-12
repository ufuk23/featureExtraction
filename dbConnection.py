import pyodbc

#conn = pyodbc.connect("DRIVER={SQL Server};SERVER=172.17.20.41;PORT=1433;UID=muesd;PWD=Mues*dev.1;DATABASE=mues_dev")
conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=172.17.20.41;PORT=1433;UID=muesd;PWD=Mues*dev.1;DATABASE=mues_dev")

cursor = conn.cursor()

cursor.execute("SELECT * FROM CONFIGURATION WHERE ID =1")
row = cursor.fetchone()

conn.close()

print(row)

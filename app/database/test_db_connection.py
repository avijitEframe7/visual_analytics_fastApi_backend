import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=192.168.0.218;"
    "DATABASE=employeeinfo;"
    "UID=sa;"
    "PWD=12345;"
    "TrustServerCertificate=yes;"
)

print("Connected Successfully!")
print(conn.getinfo(pyodbc.SQL_SERVER_NAME))
print(conn.getinfo(pyodbc.SQL_DATABASE_NAME))

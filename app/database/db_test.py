import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=192.168.0.216;"
    "DATABASE=employeeinfo;"
    "UID=sa;"
    "PWD=12345;"
    "TrustServerCertificate=yes;"
)

print("Connected Successfully!")

import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=192.168.100.103;"
    "DATABASE=employeeinfo;"
    "UID=sa;"
    "PWD=asdfg;"
    "TrustServerCertificate=yes;"
)

print("Connected Successfully!")
import pandas as pd
import pymysql
from sqlalchemy import create_engine

sqlEngine       = create_engine('mysql+pymysql://root:auth_string@localhost/statlog', pool_recycle=3600)
dbConnection    = sqlEngine.connect()
frame           = pd.read_sql("select * from statlog.germancredit", dbConnection);

pd.set_option('display.expand_frame_repr', False)

print(frame)


dbConnection.close()

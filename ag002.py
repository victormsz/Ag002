import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score


#conexão com o banco de dados mysql
sqlEngine       = create_engine('mysql+pymysql://root:auth_string@localhost/statlog', pool_recycle=3600)
dbConnection    = sqlEngine.connect()
frame           = pd.read_sql("select * from statlog.germancredit", dbConnection); #dataframe inicial


pd.set_option('display.expand_frame_repr', False) #para não truncar as colunas

frame.columns = ['Id','status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk']

X=frame.drop('credit_risk', axis=1)
y=frame['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 522 = minha matricula

# Normalização dos dados
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


MLPclf = MLPClassifier(hidden_layer_sizes=(10),
                    verbose=True,
                    learning_rate_init=0.001,
                    max_iter=10000,
                    tol=1e-4,)

MLPclf.fit(X_train,y_train)

ypred=MLPclf.predict(X_test)

# Evaluate accuracy
ypred = MLPclf.predict(X_test)
print(f"Training Accuracy: {accuracy_score(y_train, MLPclf.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, ypred)}")

# precision
print(f"Precision: {precision_score(y_test, ypred)}")
# recall
print(f"Recall: {recall_score(y_test, ypred)}")
# f1-score
print(f"F1-Score: {f1_score(y_test, ypred)}")

#scores = cross_val_score(MLPclf, X, y, cv=5)
#print(f"Cross-Validation Scores: {scores}")
#print(f"Mean Cross-Validation Accuracy: {scores.mean()}")



dbConnection.close()

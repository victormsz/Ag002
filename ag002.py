import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#conexão com o banco de dados mysql
sqlEngine       = create_engine('mysql+pymysql://root:auth_string@localhost/statlog', pool_recycle=3600)
dbConnection    = sqlEngine.connect()
frame           = pd.read_sql("select * from statlog.germancredit", dbConnection); #dataframe inicial


pd.set_option('display.expand_frame_repr', False) #para não truncar as colunas


X=frame[['laufkont','laufzeit','moral','verw','hoehe','sparkont','beszeit','rate','famges','buerge','wohnzeit','verm','alter','weitkred','wohn','bishkred','beruf','pers','telef','gastarb']]
y=frame['kredit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=522) # 522 = minha matricula

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(100,100,100),
                    verbose=True,
                    learning_rate_init=0.005,
                    max_iter=1000,)

clf.fit(X_train,y_train)

ypred=clf.predict(X_test)

# Evaluate accuracy
ypred = clf.predict(X_test)
print(f"Training Accuracy: {accuracy_score(y_train, clf.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, ypred)}")

#testing naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
ypred = gnb.predict(X_test)
print(f"Training Accuracy: {accuracy_score(y_train, gnb.predict(X_train))}")
print(f"Test Accuracy: {accuracy_score(y_test, ypred)}")


dbConnection.close()

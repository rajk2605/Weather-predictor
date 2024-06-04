#****************CONFIRMED**************(07-01-24)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/demo/ML/internship/weather_predictor/weather.csv")
print(data)
print(data.isnull().sum())

features = data[["precipitation","temp_max","temp_min","wind"]]
target = data["weather"]

#mms = MinMaxScaler()
#nfeatures = mms.fit_transform(features)

#print(features)
#print(nfeatures)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)

model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



pe = float(input("enter precipitation: "))
tmax = float(input("enter temp_max: "))
tmin = float(input("enter temp_min: "))
wind = float(input("enter wind: "))
d = [[pe, tmax, tmin, wind]]
#nd = mms.transform(d)
cl = model.predict(d)
print("The predicted Weather: ",cl)

#nn = model.kneighbors(nd, n_neighbors=5)
#print(nn)

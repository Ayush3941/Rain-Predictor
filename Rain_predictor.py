#import required modules
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#__________________________________________________________________
# :) just to ignore useless warning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
#__________________________________________________________________
file = pd.read_csv("archive/weather.csv")
file = file.dropna(subset= ["RainToday","RainTomorrow"])

#converting date column in pandas date time object
year = pd.to_datetime(file.Date).dt.year

# dividing data according to years into train test and validation set
train = file[year<2015]
val = file[year==2015]
test = file[year>2015]

# further dividing the above sets into input and target sets 
train_target_col = "RainTomorrow"
train_input_cols = list(train.columns)[1:-1]

val_target_col = "RainTomorrow"
val_input_cols = list(val.columns)[1:-1]

test_target_col = "RainTomorrow"
test_input_cols = list(test.columns)[1:-1]
#_________________________________________________________________

train_input = train[train_input_cols].copy()
val_input = val[val_input_cols].copy()
test_input = test[test_input_cols].copy()

train_target = train[train_target_col].copy()
val_target = val[val_target_col].copy()
test_target = test[test_target_col].copy()

#classifying numerical input columns from object input cols
numcols = train_input.select_dtypes(include = np.number).columns.tolist()
catcols = train_input.select_dtypes("object").columns.tolist()

#setting an imputer object to replace nan values in data 
imputer = SimpleImputer(strategy = "mean")
imputer.fit(file[numcols])

#replacing nan values in dataset
train_input[numcols] = imputer.transform(train_input[numcols])
val_input[numcols] = imputer.transform(val_input[numcols])
test_input[numcols] = imputer.transform(test_input[numcols])

#scaling the values of database in a range of 0 to 1 
#first creating a scaler object

scaler = MinMaxScaler()
scaler.fit(file[numcols])

# now assigning new values in input sets
train_input[numcols] = scaler.transform(train_input[numcols])
val_input[numcols] = scaler.transform(val_input[numcols])
test_input[numcols] = scaler.transform(test_input[numcols])

# now converting categorial columns in one hot column matrix

encoder = OneHotEncoder(sparse_output = False, handle_unknown="ignore")
encoder.fit(file[catcols])
encoded = list(encoder.get_feature_names_out(catcols))

#now assigning new columns to dataframe

train_input[encoded] = encoder.transform(train_input[catcols])
val_input[encoded] = encoder.transform(val_input[catcols])
test_input[encoded] = encoder.transform(test_input[catcols])

# now building final dataframes set to be put inside model

X_train = train_input[encoded+numcols]
X_val = val_input[encoded+numcols]
X_test = test_input[encoded+numcols]

#model making for Descision tree
#fiting the data in model
model = RandomForestClassifier(n_jobs = -1,random_state = 42,n_estimators=1000,max_depth=22,max_samples=0.45,class_weight={"No":1,"Yes":10})# random_state is used to get same type of output
model.fit(X_train,train_target)
# taking predicted values from model

train_prediction = model.predict(X_train)
val_prediction = model.predict(X_val)
test_prediction = model.predict(X_test)

# printing accuracy score for the predicted values
print(model.classes_)
print(accuracy_score(train_target,train_prediction))
print(accuracy_score(val_target,val_prediction))
print(accuracy_score(test_target,test_prediction))


# printing confusion matrix from it
a=confusion_matrix(train_target,train_prediction)
print(a)
print("chance of No to be TRUE",100*(a[0][0]/sum(a[0])),"chance of Yes to be True",100*(a[1][1]/sum(a[1])))
a = confusion_matrix(val_target,val_prediction)
print(a)
print("chance of No to be TRUE",100*(a[0][0]/sum(a[0])),"chance of Yes to be True",100*(a[1][1]/sum(a[1])))
a = confusion_matrix(test_target,test_prediction)
print(a)
print("chance of No to be TRUE",100*(a[0][0]/sum(a[0])),"chance of Yes to be True",100*(a[1][1]/sum(a[1])))


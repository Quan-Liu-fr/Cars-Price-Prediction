import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from math import sqrt

# prepare data before model training
# one-hot encoding for "Make, Model, Type, Origin, DriveTrain"
data = pd.read_csv("../input/cars_data_cleaned.csv")

data_dum = pd.get_dummies(data, columns=['Make', 'Model', 'Type', 'Origin', 'DriveTrain'])

data_train= data_dum.drop(['Invoice'],axis = 1)

# Feeding input features to X and output (MSRP) to y
X = data_train.drop('MSRP', axis = 1)
y=data_train['MSRP'] 

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state=20)

# MULTIPLE LINEAR REGRESSION MODEL
LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X_train, y_train)
filename_LR = '../model/LinearRegression_model.pickle'
pickle.dump(LinearRegression_model,open(filename_LR,'wb'))

accuracy_LinearRegression = LinearRegression_model.score(X_test, y_test)
print('LR model accuracy',accuracy_LinearRegression)

# DECISION TREE AND RANDOM FOREST MODELS
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)
filename_DT = '../model/DecisionTree_model.pickle'
pickle.dump(DecisionTree_model,open(filename_DT,'wb'))
accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
print('DT model accuracy', accuracy_DecisionTree)

RandomForest_model = RandomForestRegressor(n_estimators=5,max_depth=5)
RandomForest_model.fit(X_train, y_train)
filename_RF = '../model/RandomForest_model.pickle'
pickle.dump(RandomForest_model,open(filename_RF,'wb'))
accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
print('RF model accuracy', accuracy_RandomForest)

# XG-BOOST REGRESSOR MODEL
model = XGBRegressor()
model.fit(X_train, y_train)
filename_XG = '../model/model.pickle'
pickle.dump(model,open(filename_XG,'wb'))
accuracy_XG = model.score(X_test, y_test)
print('XG model accuracy', accuracy_XG)
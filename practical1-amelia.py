import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

use_full_train_set = 0

# Turns off scientific notation when printing matrices in numpy
np.set_printoptions(suppress=True)

# use pandas built-in functions to read in train and test
print "Loading data"
start = time.time()
df_train_all = pd.read_csv("train_1000.csv")
print "Done reading in train", time.time() - start
start = time.time()
df_test = pd.read_csv("test_1000.csv")
print "Done reading in test" , time.time() - start

# delete smiles from all data
del df_train_all['smiles']
del df_test['smiles']

# store gap values, from last column of train, as an array
Y_train_all = df_train_all.gap.values

# delete 'gap' column from train to get 256-column DF
df_train_all = df_train_all.drop(['gap'], axis=1)

# delete 'Id' column from test to get 256-column DF
df_test = df_test.drop(['Id'], axis=1)

# # remember row where testing examples will start (because we're going to concatenate them at the end of train)
# test_idx = df_train.shape[0]

# # Concatenate train & test to get a single  DataFrame, so we can more easily apply feature engineering on
# df_all = pd.concat((df_train, df_test), axis=0)

# convert DataFrames into plain arrays
X_train_all = df_train_all.values
X_test = df_test.values

if use_full_train_set == 0:
    # split training set into training and validation sets
    X_train, X_validate, Y_train, Y_validate = train_test_split(
             X_train_all, Y_train_all, test_size=0.5, random_state=0)

    print "X_train shape:", X_train.shape
    print "X_validate shape:", X_validate.shape
    print "Y_train shape:", Y_train.shape
    print "Y_validate shape:", Y_validate.shape
    print "X_test shape:", X_test.shape

else:
    X_train = X_train_all
    Y_train = Y_train_all

    print "X_train shape:", X_train.shape
    print "Y_train shape:", Y_train.shape
    print "X_test shape:", X_test.shape

def linreg(X_train, Y_train, X_validate):
    """Linear regression"""
    # create linear regression object
    LR = LinearRegression()
    # train the model using the training data
    LR.fit(X_train, Y_train)
    # print "Coefficients: \n", LR.coef_
    Y_pred = LR.predict(X_validate)
    write_to_file("LR_Y_pred.csv", Y_pred)
    return Y_pred

def simplerandomforest(X_train, Y_train, X_validate=None):
    """Simple random forest regression"""
    RF = RandomForestRegressor(verbose=2, n_estimators=50)
    RF.fit(X_train, Y_train)
    write_to_file("RF_feature_importances.csv", RF.feature_importances_)
    if use_full_train_set == 0:
        Y_pred = RF.predict(X_validate)
    else:
        Y_pred = RF.predict(X_test)
    write_to_file("RF_Y_pred.csv", Y_pred)
    return Y_pred

def svr_rbf(X_train, Y_train, X_validate):
    """Support vector regression, using RBF kernel"""
    SVR_RBF = SVR(kernel='rbf')
    SVR_RBF.fit(X_train, Y_train)
    Y_pred = SVR_RBF.predict(X_validate)
    write_to_file("SVR_RBF_Y_pred.csv", Y_pred)
    return Y_pred

def svr_linear(X_train, Y_train, X_validate):
    """Support vector regression, using linear kernel"""
    SVR_LIN = SVR(kernel='linear')
    Y_pred = SVR_LIN.fit(X_train, Y_train).predict(X_validate)
    write_to_file("SVR_LIN_Y_pred.csv", Y_pred)
    return Y_pred

def svr_poly(X_train, Y_train, X_validate):
    """Support vector regression, using polynomial kernel"""
    SVR_POLY = SVR(kernel='poly')
    Y_pred = SVR_POLY.fit(X_train, Y_train).predict(X_validate)
    write_to_file("SVR_POLY_Y_pred.csv", Y_pred)
    return Y_pred

if use_full_train_set == 0:
    start = time.time()
    Y_pred = simplerandomforest(X_train, Y_train, X_validate)
    print "Done fitting and predicting", time.time() - start

    print "Y_pred shape:", Y_pred.shape
    rmse = math.sqrt(mean_squared_error(Y_validate, Y_pred))
    print("RMSE: %.6f" % rmse)

else:
    start = time.time()
    Y_pred = simplerandomforest(X_train, Y_train)
    print "Done fitting and predicting", time.time() - start

    print "Y_pred shape:", Y_pred.shape
    # rmse = math.sqrt(mean_squared_error(Y_train, Y_pred))
    # print("RMSE: %.6f" % rmse)
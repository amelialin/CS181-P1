import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def shuffle(df):
    """
    Randomly shuffles rows of a DataFrame and returns the shuffled DataFrame
    """     
    return df.reindex(np.random.permutation(df.index))

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

# Turns off scientific notation when printing matrices in numpy
np.set_printoptions(suppress=True)

# use pandas built-in functions to read in train and test
df_train = pd.read_csv("train_1000.csv")
print "Done reading in train"
df_test = pd.read_csv("test_1000.csv")
print "Done reading in test"

# take random sample from training set
# df_train = shuffle(df_train)[:5000]

# store gap values, from last column of train
Y_train = df_train.gap.values

# delete smiles from all data
del df_train['smiles']
del df_test['smiles']

# delete 'gap' column from train to get 256-column DF
df_train = df_train.drop(['gap'], axis=1)

# delete 'Id' column from test to get 256-column DF
df_test = df_test.drop(['Id'], axis=1)

# # remember row where testing examples will start (because we're going to concatenate them at the end of train)
# test_idx = df_train.shape[0]

# # Concatenate train & test to get a single  DataFrame, so we can more easily apply feature engineering on
# df_all = pd.concat((df_train, df_test), axis=0)

# convert DataFrames into plain arrays
X_train = df_train.values
X_test = df_test.values

print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape

### create linear regression object
LR = LinearRegression()
# train the model using the training data
LR.fit(X_train, Y_train)
# print "Coefficients: \n", LR.coef_
LR_Y_pred = LR.predict(X_test)
# print "LR_Y_pred", LR_Y_pred

# ### simple random forest
# RF = RandomForestRegressor()
# RF.fit(X_train, Y_train)
# print "RF.feature_importances_", RF.feature_importances_
# RF_Y_pred = RF.predict(X_test)

### Support vector regression, using linear and non-linear, polynomial, and RBF kernels
# SVR_RBF = SVR(kernel='rbf', C=1e3, gamma=0.1)
# SVR_RBF = SVR(kernel='rbf')
# SVR_LIN = SVR(kernel='linear', C=1e3)
# SVR_POLY = SVR(kernel='poly', C=1e3, degree=2)
# SVR_RBF.fit(X_train, Y_train)
# print "Done fitting"
# SVR_RBF_Y_pred = SVR_RBF.predict(X_test)
# SVR_LIN_Y_pred = SVR_LIN.fit(X_train, Y_train).predict(X_test)
# print "Done with SVR_LIN"
# SVR_POLY_Y_pred = SVR_POLY.fit(X_train, Y_train).predict(X_test)
# print "Done with SVR_POLY"

print "Done predicting"

# write_to_file("LR_Y_pred.csv", LR_Y_pred)
# write_to_file("RF_Y_pred.csv", RF_Y_pred)
# write_to_file("Y_train.csv", Y_train)
# write_to_file("SVR_RBF_Y_pred.csv", SVR_RBF_Y_pred)
# write_to_file("SVR_LIN_Y_pred.csv", SVR_LIN_Y_pred)
# write_to_file("SVR_POLY_Y_pred.csv", SVR_POLY_Y_pred)
print "Done writing to file."

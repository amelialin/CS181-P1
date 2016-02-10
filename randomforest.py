import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Turns off scientific notation when printing matrices in numpy
np.set_printoptions(suppress=True)

"""
Read in train and test as Pandas DataFrames
"""

# use pandas built-in functions to read in train and test
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# store gap values, from last column of train
Y_train = df_train.gap.values

# delete smiles from all data
del df_train['smiles']
del df_test['smiles']

# delete 'gap' column from train to get 1000x256 DF
df_train = df_train.drop(['gap'], axis=1)

# delete 'Id' column from test to get 1000x256 DF
df_test = df_test.drop(['Id'], axis=1)

# remember row where testing examples will start (because we're going to concatenate them at the end of train)
test_idx = df_train.shape[0]

# Concatenate train & test to get a single  DataFrame, so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)

# """
# Example Feature Engineering

# this calculates the length of each smile string and adds a feature column with those lengths
# Note: this is NOT a good feature and will result in a lower score!
# """
# # create column of new values
# smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
# # appends new column to end of df_all with name 'smiles_len'
# df_all['smiles_len'] = pd.DataFrame(smiles_len)
# print df_all.head

# convert DataFrame into plain array
vals = df_all.values

# split it back into 2 separate arrays for test and train, with 1000 rows each
X_train = vals[:test_idx]
X_test = vals[test_idx:]

print "Train features:", X_train.shape
print "Train gap:", Y_train.shape
print "Test features:", X_test.shape

# ### create linear regression object
# LR = LinearRegression()
# # train the model using the training data
# LR.fit(X_train, Y_train)
# # print "Coefficients: \n", LR.coef_
# LR_Y_pred = LR.predict(X_test)
# # print "LR_Y_pred", LR_Y_pred

# ### simple random forest
# RF = RandomForestRegressor()
# RF.fit(X_train, Y_train)
# print "RF.feature_importances_", RF.feature_importances_
# RF_Y_pred = RF.predict(X_test)

### Support vector regression, using linear and non-linear, polynomial, and RBF kernels
# SVR_RBF = SVR(kernel='rbf', C=1e3, gamma=0.1)
SVR_RBF = SVR(kernel='rbf')
# SVR_LIN = SVR(kernel='linear', C=1e3)
# SVR_POLY = SVR(kernel='poly', C=1e3, degree=2)
SVR_RBF_Y_pred = SVR_RBF.fit(X_train, Y_train).predict(X_test)
print "Done with SVR_RBF"
# SVR_LIN_Y_pred = SVR_LIN.fit(X_train, Y_train).predict(X_test)
# print "Done with SVR_LIN"
# SVR_POLY_Y_pred = SVR_POLY.fit(X_train, Y_train).predict(X_test)
# print "Done with SVR_POLY"

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

# write_to_file("LR_Y_pred.csv", LR_Y_pred)
# write_to_file("RF_Y_pred.csv", RF_Y_pred)
# write_to_file("Y_train.csv", Y_train)
write_to_file("SVR_RBF_Y_pred.csv", SVR_RBF_Y_pred)
# write_to_file("SVR_LIN_Y_pred.csv", SVR_LIN_Y_pred)
# write_to_file("SVR_POLY_Y_pred.csv", SVR_POLY_Y_pred)

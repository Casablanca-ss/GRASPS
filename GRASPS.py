from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from scipy.stats import pearsonr
import pandas as pd

# load data
data_train = pd.read_csv(
    r".\csv\all_data_train_del_abnormal.csv")
data_test = pd.read_csv(
    r".\csv\all_data_test_del_abnormal.csv")

# choice input variables
predictors_train = data_train[["gpm", "trmm", "cdr", "cloud", "month", "day2"]].values
predictors_test = data_test[["gpm", "trmm", "cdr", "cloud", "month", "day2"]].values

# choice output variable
Y_train = data_train["std"].values
Y_test = data_test["std"].values

# normalization
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(predictors_train)  # training set normalization
X_test = min_max_scaler.transform(predictors_test)  # test set normalization

# load RF model
regressor = RandomForestRegressor(n_estimators=200, max_depth=20)

# training
regressor.fit(X_train, Y_train)

# save model
joblib.dump(regressor, r"gpm, trmm, cdr,cloud, month,day2.joblib")
# use model
regressor = joblib.load(r"gpm, trmm, cdr,cloud, month,day2.joblib")

Y_pred = regressor.predict(X_test)  # predict test set
Y_pred2 = regressor.predict(X_train)  # predict training set

# model evaluation
print('Done.\ntrain:\nR-squared: %f\nMSE: %f\n' % (r2_score(Y_train, Y_pred2), mean_squared_error(Y_train, Y_pred2)))
print('Done.\ntest:\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
print('RMSE: %f' % (mean_squared_error(Y_test, Y_pred, squared=False)))
print('MAE: %f' % (mean_absolute_error(Y_test, Y_pred)))
print('pearson:', pearsonr(Y_pred, Y_test))

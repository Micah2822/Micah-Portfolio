

from Fight_Prediction_Model.data_processing.data_select_clean import x_train, x_test, x_cv, y_cv, y_test, y_train, dataset
from Fight_Prediction_Model.NN_ML_Model.nn_model import model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

x_cv = tf.convert_to_tensor(x_cv, dtype=tf.float32)
y_cv = tf.convert_to_tensor(y_cv, dtype=tf.float32)

# Compute the test MSE
yhat_test = model.predict(x_test)
test_mse = mean_squared_error(y_test, yhat_test) / 2
print("Test MSE: ", test_mse)

# Record the training MSE
yhat_training = model.predict(x_train)
train_mse = mean_squared_error(y_train, yhat_training) / 2
print("Train MSE: ", train_mse)

# Record the cross validation MSEs
yhat_cv = model.predict(x_cv)
cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
print("CV MSE: ", cv_mse)


from Fight_Prediction_Model.data_processing.data_select_clean import x_train, x_test, x_cv, y_cv, y_test, y_train
from Fight_Prediction_Model.NN_ML_Model.nn_model import model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

x_cv = tf.convert_to_tensor(x_cv, dtype=tf.float32)
y_cv = tf.convert_to_tensor(y_cv, dtype=tf.float32)

# Predict using the model
yhat_test = model.predict(x_test).flatten()
yhat_test = [1 if y > 0.5 else 0 for y in yhat_test]  # Convert probabilities to class labels

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, yhat_test)
precision = precision_score(y_test, yhat_test)
recall = recall_score(y_test, yhat_test)
f1 = f1_score(y_test, yhat_test)
roc_auc = roc_auc_score(y_test, yhat_test)

# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

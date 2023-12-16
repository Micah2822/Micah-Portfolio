import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
# from tensorflow.keras import SparseCategoricalCrossentropy
from Fight_Prediction_Model.data_processing.data_select_clean import x_train, y_cv, y_test, y_train, dataset

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Number of input features
n_features = x_train.shape[1]

# Create model
model = Sequential([
    #Dense(units=20,  input_dim=n_features, activation='relu'),
    Dense(units=20, activation='relu'),
    Dense(units=10, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.compile(loss=SparseCategoricalCrossentropy() )
# model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Step 3: Train data to minimise cost function:
model.fit(x_train, y_train, epochs=100)

# Summary of the model
model.summary()



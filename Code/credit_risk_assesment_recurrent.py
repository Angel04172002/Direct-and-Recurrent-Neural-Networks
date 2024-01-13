import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
num_samples = 1000
sequence_length = 10

X = np.random.rand(num_samples, sequence_length, 5)

y = np.random.randint(2, size=(num_samples,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, X_train.shape[-1]), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

new_data = np.random.rand(5, sequence_length, 5)
scaled_new_data = scaler.transform(new_data.reshape(-1, new_data.shape[-1])).reshape(new_data.shape)
predictions = model.predict(scaled_new_data)

print('Predictions:')
print(predictions)

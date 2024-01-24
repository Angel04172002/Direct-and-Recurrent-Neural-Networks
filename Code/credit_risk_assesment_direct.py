import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load Excel data
excel_data = pd.read_excel('./credit-risk-data.xlsx')

# Define features and target variable
X_columns = ['Borrower ID', 'Credit History Absences', 'Credit History Loan Sum ', 'Monthly Income ',
             'Current Liabilities ', 'Num Credit Cards ', 'Age', 'Education', 'Marital Status']

y_column = 'Default Status'

# Separate numerical and categorical columns
numeric_columns = excel_data[X_columns].select_dtypes(include=['float64', 'int64']).columns
categorical_columns = excel_data[X_columns].select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent for categorical data
    ('onehot', OneHotEncoder())
])

# Use ColumnTransformer for handling both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Apply preprocessing and create the final feature matrix X and target variable y
X = preprocessor.fit_transform(excel_data[X_columns])
y = excel_data[y_column].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Generate predictions for new data
new_data = np.random.rand(5, X_train.shape[1])
predictions = model.predict(new_data)

print('Predictions:')
print(predictions)


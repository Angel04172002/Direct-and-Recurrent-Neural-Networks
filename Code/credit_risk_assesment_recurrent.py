import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# Load Excel data
excel_data = pd.read_excel('./credit-risk-data.xlsx')


print(excel_data)

# Define features and target variable
X_columns = ['Borrower ID', 'Credit History Absences', 'Credit History Loan Sum ', 'Monthly Income ',
             'Current Liabilities ', 'Num Credit Cards ', 'Age', 'Education', 'Marital Status']

y_column = 'Default Status'

# Separate numerical and categorical columns
numeric_columns = excel_data[X_columns].select_dtypes(include=['float64', 'int64']).columns
categorical_columns = excel_data[X_columns].select_dtypes(include=['object']).columns

# Impute NaN values in numerical columns
numeric_imputer = SimpleImputer(strategy='mean')
excel_data[numeric_columns] = numeric_imputer.fit_transform(excel_data[numeric_columns])

# Check if there are any NaN values in the data
if excel_data.isna().any().any():
    # Handle NaN values or drop rows as needed
    excel_data = excel_data.dropna()  # Example: Drop rows with NaN values

# Check if there are any samples left in the data
if excel_data.shape[0] > 0:
    # Use ColumnTransformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # Apply the transformations
    X_transformed = preprocessor.fit_transform(excel_data[X_columns])

    # Concatenate the transformed data with the target variable
    X = np.concatenate((X_transformed, excel_data[y_column].values.reshape(-1, 1)), axis=1)
    y = excel_data[y_column].values

    # Ensure that X and y have the same number of samples
    y = y[:X.shape[0]]

    # Flatten y
    y = y.flatten()

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
else:
    print("No samples left after preprocessing.")

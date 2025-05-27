import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Extract - Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data

# 2. Transform - Preprocessing pipeline
def preprocess_data(data):
    # Separate features by type
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),    # Fill missing numeric values with mean
        ('scaler', StandardScaler())                     # Scale numeric features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical features
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(data)

    # Convert the result back to a DataFrame with proper column names
    # Extract feature names after OneHotEncoding
    cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_cols = numeric_features + list(cat_cols)

    df_transformed = pd.DataFrame(transformed_data.toarray() if hasattr(transformed_data, "toarray") else transformed_data,
                                  columns=all_cols)

    print("Data preprocessing complete.")
    return df_transformed

# 3. Load - Save the processed data
def save_data(data, output_path):
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

# Main pipeline execution
if __name__ == "__main__":
    input_file = 'input_data.csv'     # Replace with your input CSV path
    output_file = 'processed_data.csv'  # Replace with your desired output path

    # Run ETL steps
    raw_data = load_data(input_file)
    processed_data = preprocess_data(raw_data)
    save_data(processed_data, output_file)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def extract(file_path):
    """Extract data from CSV file"""
    data = pd.read_csv(file_path)
    print("Data extracted")
    return data

def transform(data):
    """Preprocess and transform the data"""

    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Preprocessing for numerical data: just impute missing values (mean)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Preprocessing for categorical data: impute missing with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit and transform data
    data_processed = preprocessor.fit_transform(data)

    print("Data transformed")

    # Get feature names
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(cat_features)

    # Convert to DataFrame
    data_processed_df = pd.DataFrame(
        data_processed.toarray() if hasattr(data_processed, "toarray") else data_processed,
        columns=all_features
    )

    return data_processed_df

def load(data, output_file):
    """Load the processed data to CSV"""
    data.to_csv(output_file, index=False)
    print(f"Data loaded to {output_file}")

def etl_pipeline(input_file, output_file):
    data = extract(input_file)
    processed_data = transform(data)
    load(processed_data, output_file)

if __name__ == "__main__":
    input_file = "input_data.csv"  # Your input CSV filename
    output_file = "processed_data.csv"  # Output CSV filename
    etl_pipeline(input_file, output_file)

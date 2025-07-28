import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    This function loads the Eagle dataset from the given file path.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        DataFrame : The loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path, compression='bz2')

def filter_data(df):
    """
    Filter the dataset into successful and failed jobs.

    Args:
        df (DataFrame): The dataset to be filtered.

    Returns:
        df_success, df_failure (DataFrame,DataFrame): The filtered datasets for successful and failed jobs.
    """
    df_success0 = df[df['state'] == 'COMPLETED']
    df_success = df_success0[df_success0['run_time'] != 0]
    df_failure = df[df['state'] == 'TIMEOUT']
    return df_success, df_failure

def calculate_wait_time(df):
    """
    This function calculates the wait time for successful jobs.

    Args:
        df (DataFrame): The dataset containing successful jobs.

    Returns:
        df (DataFrame): The dataset with the wait time calculated.
    """
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['submit_time'] = pd.to_datetime(df['submit_time'])
    df['wait_time'] = (df['start_time'] - df['submit_time']).dt.total_seconds()
    return df

def encode_categorical_features(df, categorical_features):
    """
    This function encodes the categorical features in the dataset

    Args:
        df (DataFrame): The dataset to be encoded.
        categorical_features (list): The list of categorical features to be encoded.

    Returns:
        df (dataFrame): The dataset with the categorical features encoded.
    """
    label_encoders = {}
    for col in categorical_features:
        print(f'Encoding the feature {col}')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df

def preprocess_data(file_path):
    """
    This function preprocesses the Eagle dataset.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        df_success, df_failure (DataFrame, DataFrame): The preprocessed datasets for successful and failed jobs.
    """
    print("Loading data...\n")
    df = load_data(file_path)
    
    print("Filtering data...\n")
    df_success, df_failure = filter_data(df)
    
    print("Calculating wait time for successful jobs...\n")
    df_success = calculate_wait_time(df_success)
    
    print("Encoding categorical features...\n")
    categorical_features = ['job_id', 'user', 'account', 'partition', 'qos', 'name', 'work_dir', 'submit_line']
    df_success = encode_categorical_features(df_success, categorical_features)
    
    print("Preprocessing complete!\n")
    return df_success, df_failure
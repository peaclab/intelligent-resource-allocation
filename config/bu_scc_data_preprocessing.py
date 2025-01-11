import pandas as pd
import numpy as np
import datetime
import re
from sklearn.preprocessing import LabelEncoder

def load_and_filter_data(file_path, col_headers):
    """
    This function loads data from a CSV file and applies initial filtering.

    Args:
        file_path (str): File path to the CSV file.
        col_headers (list): Column headers for the DataFrame.

    Returns:
        DataFrame: Filtered DataFrame.
    """

    df = pd.read_csv(file_path, encoding='utf-8', skiprows=4, names=col_headers, sep=':')
    df = df[df['ru_wallclock'] != 0]
    df = df[df['failed'] == 0]
    df = df[df['exit_status'] == 0]
    df['lag_time'] = df['start_time'] - df['submission_time']
    df = df[df['lag_time'] > 0]
    return df.dropna()

def calculate_additional_features(df):
    """
    This function calculates additional features based on existing columns.

    Returns:
       DataFrame: DataFrame with additional features.
    """

    df['execution_time'] = df['end_time'] - df['start_time']
    df['total_cpu_time'] = df['ru_utime'] + df['ru_stime']
    df['ncpu'] = np.ceil(df['total_cpu_time'] / df['ru_wallclock'])
    df['cpu_waste'] = df['slots'] - df['ncpu']
    df['submission_time'] = df['submission_time'].apply(
        lambda x: datetime.datetime.fromtimestamp(x)
    )
    return df

def drop_unused_columns(df):
    """
    This function drops columns that are not required for the analysis.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with unused columns removed.
    """
    return df.drop(['account', 'job_number', 'failed', 'exit_status', 'iow', 
                    'pe_taskid', 'arid', 'ar_submission_time'], axis=1)


def parse_h_rt_flag(column):
    """
    This function parses the h_rt (hard runtime) flag from the category column.

    Args:
        column (str): Category column contains the flags passed to the job scheduler.

    Returns:
        parsed_data (list): List of dictionaries containing the parsed h_rt values.
    """
    parsed_data = []
    for entry in column:
        matches = re.findall(r'-l ([^\s-]+)', entry)
        h_rt_value = None
        for match in matches:
            for item in match.split(','):
                if item.startswith('h_rt='):
                    _, value = item.split('=', 1)
                    h_rt_value = int(value)
        parsed_data.append({'h_rt': h_rt_value})
    return parsed_data

def add_h_rt_column(df):
    """
    This function parses the h_rt flag from the category column and adds a new column to the DataFrame.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        df (DataFrame): DataFrame with the h_rt (hard runtime) column added.
    """

    parsed_flags = parse_h_rt_flag(df['category'])
    parsed_df = pd.DataFrame(parsed_flags)
    df = pd.concat([df, parsed_df], axis=1)
    df['h_rt'] = df['h_rt'].fillna(12 * 60 * 60).astype(int)
    return df

def encode_categorical_features(df, categorical_features):
    """
    This function encodes categorical features using LabelEncoder.

    Args:
        df (DataFrame): Input DataFrame.
        categorical_features (list): List of categorical features to encode.

    Returns:
        df (DataFrame): DataFrame with encoded categorical features.
    """
    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        df[col] = label_encoders[col].fit_transform(df[col])
    return df

def preprocess_data(file_path, col_headers):
    """
    This function preprocesses the data by loading, filtering, and encoding features.

    Args:
        file_path (str): Path to the input file.
        col_headers (list): List of column headers.

    Returns:
        df (DataFrame): Preprocessed DataFrame.
    """
    print("Loading and filtering data...\n")
    df = load_and_filter_data(file_path, col_headers)
    
    print("Calculating additional features...\n")
    df = calculate_additional_features(df)
    
    print("Dropping unused columns...\n")
    df = drop_unused_columns(df)
    
    print("Parsing h_rt flags...\n")
    df = add_h_rt_column(df)
    print("Parsing done!\n")
    
    print("Encoding categorical features...\n")
    categorical_submission_features = ['group', 'owner', 'job_name', 'department', 'granted_pe']
    df = encode_categorical_features(df, categorical_submission_features)
    print("Encoding done!\n")
    
    return df

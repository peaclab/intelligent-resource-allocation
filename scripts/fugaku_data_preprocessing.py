import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(directory):
    """
    This function loads the dataset from the specified Parquet file.
    """
    return pd.read_parquet(directory)

def convert_to_datetime(df, datetime_columns):
    """
    This function converts the specified columns to datetime format.

    Args:
        df (DataFrame): The input DataFrame.
        datetime_columns (list): List of column names to convert.

    Returns:
       DataFrame: The DataFrame with updated datetime columns.
    """

    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])
    return df

def calculate_time_metrics(df):
    """
    This function calculates the wait time and run time in seconds.

    Args:
        df (DataFrame): The nput DataFrame.

    Returns:
        df (DataFrame): The DataFrame with calculated time metrics.
    """

    df['wait_time'] = (df['sdt'] - df['adt']).dt.total_seconds()
    df['run_time'] = (df['edt'] - df['sdt']).dt.total_seconds()
    return df

def split_by_exit_state(df):
    """
    This function splits the dataframe into successful and failed jobs based on the exit state.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        df_success, df_failure (DataFrame, DataFrame): Two DataFrames, one for successful jobs and one for failed jobs.
    """
    df_success0 = df[df['exit state'] == 'completed']
    df_success = df_success0[df_success0['run_time'] != 0]
    df_failure = df[df['exit state'] == 'failed']
    return df_success, df_failure

def sort_by_column(df, column_name):
    """
    This function sorts the DataFrame (df) by a specified column.

    Args:
        df (DataFrame): The input DataFrame.
        column_name (str): The column to sort by.

    Returns:
        DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by=column_name)

def encode_categorical_features(df, categorical_features):
    """
    This function encodes specified categorical features using Label Encoding.
    
    Args:
        df (DataFrame): The input DataFrame.
        categorical_features (list): List of categorical feature column names.

    Returns:
       df(DataFrame): The DataFrame with encoded features.
    """
    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        df[col] = label_encoders[col].fit_transform(df[col])
    return df


def preprocess_data(directory):
    """
    The main data preprocessing function for Fugaku dataset.

    Args:
        directory (str): The path to the dataset file.

    Returns:
        df_success, df_failure, numerical_submission_features (DataFrame, DataFrame, list): Preprocessed successful jobs DataFrame, failed jobs DataFrame, numerical features list.
    """
    df = load_dataset(directory)
    df = convert_to_datetime(df, ['adt', 'sdt', 'edt'])
    df = calculate_time_metrics(df)
    sorted_df = sort_by_column(df, 'edt')
    df_success, df_failure = split_by_exit_state(sorted_df)

    numerical_submission_features = ['cnumr', 'nnumr', 'elpl', 'mszl', 'freq_req']
    categorical_submission_features = ['jid', 'usr', 'jnam', 'jobenv_req']

    df_success = encode_categorical_features(df_success, categorical_submission_features)

    return df_success, df_failure, numerical_submission_features


def convert_fugaku_to_swf(df):
    """
    This function converts the Fugaku dataset to the Standard Workload Format (SWF).

    Args:
        df (DataFrame): The input DataFrame containing Fugaku job data.

    Returns:
        DataFrame: The DataFrame converted to SWF format.
    """
    swf_df = pd.DataFrame()
    
    df = df.sort_values(by='adt').reset_index(drop=True)
    swf_df['Job Number'] = range(1, len(df) + 1)

    
    df['adt'] = pd.to_datetime(df['adt'], errors='coerce') 
    df['adt'] = df['adt'].astype('int64') // 10**9  
    min_submit_time = df['adt'].min()
    swf_df['Submit Time'] = df['adt'] - min_submit_time
    

    swf_df['Wait Time'] = df['wait_time'].fillna(-1)
    swf_df['Run Time'] = df['run_time'].fillna(-1)
    

    swf_df['Number of Allocated Processors'] = df['cnumat'].fillna(-1)
    

    swf_df['Average CPU Time Used'] = (df['usctmut'].fillna(0) / df['cnumat'].replace(0, np.nan)).fillna(-1)
    
    swf_df['Used Memory'] = df['mmszu'].fillna(-1)
    swf_df['Requested Number of Processors'] = df['cnumr'].fillna(-1)
    swf_df['Requested Time'] = df['elpl'].fillna(-1)
    swf_df['Requested Memory'] = df['mszl'].fillna(-1)
    
    status_mapping = {'completed': 1, 'FAILED': 0, 'CANCELLED': 5}
    swf_df['Status'] = df['exit state'].map(status_mapping).fillna(-1)
    
    user_mapping = {user: idx + 1 for idx, user in enumerate(df['usr'].unique())}
    swf_df['User ID'] = df['usr'].map(user_mapping).fillna(-1)
    
    swf_df['Group ID'] = -1
    
    # Executable (Application) Number (mapping job names to IDs)
    app_mapping = {app: idx + 1 for idx, app in enumerate(df['jnam'].unique())}
    swf_df['Executable (Application) Number'] = df['jnam'].map(app_mapping).fillna(-1)
    

    swf_df['Queue Number'] = -1  # No info
    swf_df['Partition Number'] = -1  # No info
    swf_df['Preceding Job Number'] = -1  # No info
    swf_df['Think Time from Preceding Job'] = -1  # No info
    
    return swf_df    
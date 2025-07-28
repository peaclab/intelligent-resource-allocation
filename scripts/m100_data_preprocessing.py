import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import timedelta


def convert_time_limit_to_seconds(time_limit_str):
    """
    This function converts a time limit string to seconds.

    Args:
        time_limit_str (str): The time limit string to be converted.

    Raises:
        ValueError: The time limit string is not in the correct format.
        ValueError: The time limit string is not in the correct format.

    Returns:
        total_seconds (int) : Total seconds of the time limit.
    """
    if '-' in time_limit_str:
        days, time_part = time_limit_str.split('-')
        time_parts = time_part.split(':')

        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours, minutes = map(int, time_parts)
            seconds = 0
        else:
            raise ValueError(f"Invalid time format: {time_limit_str}")

        total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    else:
        time_parts = time_limit_str.split(':')

        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours, minutes = map(int, time_parts)
            seconds = 0
        else:
            raise ValueError(f"Invalid time format: {time_limit_str}")

        total_seconds = hours * 3600 + minutes * 60 + seconds

    return total_seconds


def preprocess_data(directory):
    """
    This function preprocesses the data from a parquet file.

    Args:
        directory (str): File path to the parquet file.

    Returns:
        df_success, numerical_submission_features (DataFrame, list): The processed dataframe and the numerical features.
    """
    
    df = pd.read_parquet(directory)

    # Convert time columns to datetime format
    df['submit_time'] = pd.to_datetime(df['submit_time'], format='%Y:%m:%d %H')
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y:%m:%d %H')
    df['end_time'] = pd.to_datetime(df['end_time'], format='%Y:%m:%d %H')

    # Calculate wait time and execution time
    df['wait_time'] = (df['start_time'] - df['submit_time']).dt.total_seconds()
    df['execution_time'] = (df['end_time'] - df['start_time']).dt.total_seconds()

    # Convert time limit strings to seconds
    df['time_limit_sec'] = df['time_limit_str'].apply(convert_time_limit_to_seconds)

    # Filter for successful jobs
    success_df0 = df[df['job_state'] == 'COMPLETED']
    success_df = success_df0[success_df0['execution_time'] != 0 ]

    # Select relevant columns
    cleaned_df = success_df[['array_job_id', 'cpus_alloc_layout', 'end_time', 'group_id', 
                              'job_id', 'nodes', 'num_cpus', 'num_nodes', 'partition', 'priority',
                              'qos', 'start_time', 'submit_time', 'time_limit_sec', 'user_id', 
                              'wait_time', 'execution_time']]

    # Sort by start time and reset index
    df_success = cleaned_df.sort_values(by=['start_time']).reset_index(drop=True)

    # Convert categorical columns to integer type
    df_success['partition'] = df_success['partition'].astype(int)
    df_success['qos'] = df_success['qos'].astype(int)

    # Define numerical features for submission
    numerical_submission_features = ['array_job_id', 'group_id', 'job_id', 'partition', 
                                      'priority', 'qos', 'user_id', 'time_limit_sec']

    return df_success, numerical_submission_features

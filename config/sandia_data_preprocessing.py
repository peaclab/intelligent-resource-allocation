import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path, drop_columns, categorical_features):
    df = pd.read_csv(file_path)
    df = df.drop(drop_columns, axis=1)

    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        df[col] = label_encoders[col].fit_transform(df[col])

    def time_to_seconds(time_str):
        # Check if the string contains days (e.g., '1-16:30:45')
        if '-' in time_str:
            days, time_part = time_str.split('-')
            h, m, s = map(int, time_part.split(':'))
            return int(days) * 86400 + h * 3600 + m * 60 + s  # Convert days to seconds
        else:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

    df['execution_time'] = df['Elapsed'].apply(time_to_seconds)
    df['start_time'] = pd.to_datetime(df['Start'], unit='s')
    
    return df
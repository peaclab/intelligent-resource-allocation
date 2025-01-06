import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def cluster_create_sub_dataframes(df, selected_features, target_feature, n_clusters):

    if not all(feature in df.columns for feature in selected_features):
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        raise ValueError(f"The following features are not in the DataFrame: {missing_features}")
    

    feature_data = df[selected_features]
    feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if feature_data.isnull().any().any():
        feature_data.fillna(feature_data.mean(), inplace=True)  # Fill NaN with column means
    
    if not all(pd.api.types.is_numeric_dtype(feature_data[col]) for col in feature_data.columns):
        raise ValueError("All selected features must be numeric for clustering.")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    feature_data_scaled = feature_data.values 
    df['cluster'] = kmeans.fit_predict(feature_data_scaled)
    
    sub_dataframes = [df[df['cluster'] == cluster].drop(columns=['cluster']) for cluster in range(n_clusters)]
    
    return sub_dataframes
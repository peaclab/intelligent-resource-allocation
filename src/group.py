#!/share/pkg.8/python3/3.12.4/install/bin/python3

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def create_sub_dataframes(self, df, selected_features, days_to_train=None):
        if days_to_train is not None:
            df = df.tail(days_to_train)
        
        feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean(numeric_only=True))
        feature_data_scaled = feature_data.values

        df['cluster'] = self.kmeans.fit_predict(feature_data_scaled)
        
        cluster_centers = self.kmeans.cluster_centers_
        
        sub_dataframes = [df[df['cluster'] == cluster].drop(columns=['cluster']) for cluster in range(self.n_clusters)]
        
        return sub_dataframes, cluster_centers

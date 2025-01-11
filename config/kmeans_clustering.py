import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



def find_optimal_clusters_and_plot(df, selected_features, filepath, max_clusters=10):
    if not all(feature in df.columns for feature in selected_features):
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        raise ValueError(f"The following features are not in the DataFrame: {missing_features}")
    
    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    
    if not all(pd.api.types.is_numeric_dtype(feature_data[col]) for col in feature_data.columns):
        raise ValueError("All selected features must be numeric for clustering.")
    
    feature_data_scaled = feature_data.values 

    wcss = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(feature_data_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(feature_data_scaled, labels))
    
    # Plot WCSS and Silhouette Score
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(range(2, max_clusters + 1), wcss, marker='o', label='WCSS')
    ax[0].set_title('Elbow Method')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('WCSS')
    ax[0].legend()

    ax[1].plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='orange', label='Silhouette Score')
    ax[1].set_title('Silhouette Score')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('Score')
    ax[1].legend()

    plot_name = 'cluster_analysis.png'
    file_path = os.path.join(filepath, plot_name)
    plt.savefig(file_path)

    plt.show()
    
    # Determine optimal number of clusters using the "elbow" point for WCSS
    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)
    optimal_clusters_wcss = np.argmax(wcss_diff2 < 0) + 2  # First point where curvature changes significantly
    
    # Determine optimal number of clusters using Silhouette Score
    optimal_clusters_silhouette = np.argmax(silhouette_scores) + 2
    
    print(f"Optimal number of clusters according to Elbow Method: {optimal_clusters_wcss}")
    print(f"Optimal number of clusters according to Silhouette Score: {optimal_clusters_silhouette}")
    
    return optimal_clusters_wcss, optimal_clusters_silhouette


def cluster_create_sub_dataframes(df, selected_features, target_features, filepath, n_clusters):
    if not all(feature in df.columns for feature in selected_features):
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        raise ValueError(f"The following features are not in the DataFrame: {missing_features}")
    
    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    feature_data_scaled = feature_data.values  # Use scaled/normalized features if required

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(feature_data_scaled)
    
    # Plot additional features
    if target_features:
        for feature in target_features:
            if feature in df.columns:
                plt.figure(figsize=(8, 6))
                for cluster in range(n_clusters):
                    cluster_data = df[df['cluster'] == cluster]
                    plt.plot(cluster_data.index, cluster_data[feature], label=f'Cluster {cluster}')
                plt.title(f'{feature} per Cluster')
                plt.xlabel('Index')
                plt.ylabel(feature)
                plt.legend()

                plot_name = f'{feature}_per_cluster.png'
                file_path = os.path.join(filepath, plot_name)
                plt.savefig(file_path)

                plt.show()
            else:
                print(f"Feature {feature} not found in DataFrame. Skipping plot.")
    
    sub_dataframes = [df[df['cluster'] == cluster].drop(columns=['cluster']) for cluster in range(n_clusters)]
    
    return sub_dataframes, kmeans
 
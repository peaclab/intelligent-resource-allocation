import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def create_sub_dataframes(df, selected_features, n_clusters):

    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean(numeric_only=True))
    feature_data_scaled = feature_data.values  

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    df['cluster'] = kmeans.fit_predict(feature_data_scaled)
    
    cluster_centers = kmeans.cluster_centers_
    
    sub_dataframes = [df[df['cluster'] == cluster].drop(columns=['cluster']) for cluster in range(n_clusters)]
    
    return sub_dataframes, cluster_centers


def calculate_and_plot_wcss(df, selected_features, filepath, max_clusters=10):
    """
    This function calculates the Within-Cluster Sum of Squares (WCSS) for a range of cluster numbers and plots the results.

    Args:
        df (DataFrame): The input DataFrame.
        selected_features (list): A list of features to use for clustering.
        filepath (str): The path to save the plot.
        max_clusters (int, optional): The number of clusters to test. Defaults to 10.

    Returns:
        list: A list of WCSS values for each cluster number.
    """
    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean(numeric_only=True))
    scaler = MinMaxScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    wcss_values = []  
    
    for k in range(2, max_clusters + 1):
        print(f'Cluster = {k}\n')
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(feature_data_scaled)
        centroids = kmeans.cluster_centers_

        wcss = 0
        for cluster in np.unique(df['cluster']):
            cluster_points = feature_data_scaled[df['cluster'] == cluster]
            centroid = centroids[cluster]
            distances = cluster_points - centroid
            squared_distances = np.sum(distances**2, axis=1)
            wcss += np.sum(squared_distances)
        
        wcss_values.append(wcss)
        print(f"WCSS for k={k}: {wcss}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), wcss_values, marker='o', color= 'orange' , label='WCSS')
    plt.title('Elbow Method for M100 CINECA Dataset')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS Score')
    plt.legend()
    
    plot_name = 'wcss_per_k_plot.png'
    file_path = os.path.join(filepath, plot_name)
    plt.savefig(file_path)
    
    plt.show()
    
    return wcss_values

def calculate_silhouette_scores(df, selected_features, filepath, max_clusters=10):
    """
    This function calculates the Silhouette Score for a range of cluster numbers and plots the results.

    Args:
        df (DataFrame): The input DataFrame.
        selected_features (list): A list of features to use for clustering.
        filepath (str): The path to save the plot.
        max_clusters (int, optional): The number of clusters to test. Defaults to 10.

    Returns:
        list: A list of WCSS values for each cluster number.
    """
    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean(numeric_only=True))
    scaler = MinMaxScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    silhouette_scores = []  

    for k in range(2, max_clusters + 1):
        print(f'Cluster = {k}\n')
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(feature_data_scaled)
        
        score = silhouette_score(feature_data_scaled, labels)
        silhouette_scores.append(score)
        print(f"Silhouette Score for k={k}: {score}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='blue', label='Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.legend()

    plot_name = 'silhouette_per_k_plot.png'
    file_path = os.path.join(filepath, plot_name)
    plt.savefig(file_path)

    plt.show()
    
    return silhouette_scores


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


def cluster_create_sub_dataframes(df, selected_features, n_clusters):
    """
    This function creates sub-dataframes for each cluster and returns a list of these dataframes.

    Args:
        df (DataFrame): The input DataFrame.
        selected_features (list): A list of features to use for clustering.
        n_clusters (int): The number of clusters to create.

    Raises:
        ValueError: Raises if the selected features are not in the DataFrame.

    Returns:
        sub_dataframes, cluster_centers : The list of sub-dataframes and the cluster centers.
    """

    if not all(feature in df.columns for feature in selected_features):
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        raise ValueError(f"The following features are not in the DataFrame: {missing_features}")
    
    feature_data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(df.mean(numeric_only=True))
    feature_data_scaled = feature_data.values  

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    df['cluster'] = kmeans.fit_predict(feature_data_scaled)
    cluster_centers = kmeans.cluster_centers_
    
    sub_dataframes = [df[df['cluster'] == cluster].drop(columns=['cluster']) for cluster in range(n_clusters)]
    
    return sub_dataframes, cluster_centers
 
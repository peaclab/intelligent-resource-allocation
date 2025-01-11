from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def xgboost_feature_selection(features, pred_feature, df, title, xlabel, minR, maxR):
    X = df[features]
    y = df[pred_feature]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    rmse_values = []
    combinations = []
    best_rmse = float('inf')
    best_combination = None

    for r in range(minR, maxR + 1):
        #print(f'Iteration: {r}')

        for combo in itertools.combinations(features, r):
            #print(f'Feature Combination: {list(combo)}')

            # Subset features
            X_train_combo = X_train[list(combo)]
            X_val_combo = X_val[list(combo)]

            # Train the model
            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
            model.fit(X_train_combo, y_train)

            # Predict and calculate RMSE
            y_pred = model.predict(X_val_combo)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Track the best combination
            if rmse < best_rmse:
                best_rmse = rmse
                best_combination = combo

            # Append current combination and its RMSE
            rmse_values.append(rmse)
            combinations.append(', '.join(combo))

    # After all combinations are processed, sort and plot
    sorted_indices = np.argsort(rmse_values)[::-1]
    sorted_rmse_values = np.array(rmse_values)[sorted_indices]
    sorted_combinations = np.array(combinations)[sorted_indices]

    norm = plt.Normalize(vmin=min(sorted_rmse_values), vmax=max(sorted_rmse_values))
    colors = cm.viridis(norm(sorted_rmse_values))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_combinations, sorted_rmse_values, color=colors)

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'))
    cbar.set_label('RMSE', fontsize=12)

    plt.xlabel(f'{xlabel}', fontsize=14)
    plt.ylabel('Feature Combination', fontsize=14)
    plt.yticks(rotation=45)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=14)
    plt.title(f'{title}', fontsize=16)
    plt.tight_layout()
    plt.show()

    return best_combination


def random_forest_feature_importance(features, pred_feature, df, top_k=5):
    X = df[features]
    y = df[pred_feature]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    
    '''
    print("Feature Importance (Random Forest):")
    for idx in sorted_indices:
        print(f"{features[idx]}: {importance[idx]:.4f}")
    '''    
    
    top_features = [features[idx] for idx in sorted_indices[:top_k]]
    print(f"\nTop features based on importance: {top_features}")
    
    X_train_top = X_train[top_features]
    X_val_top = X_val[top_features]
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_top, y_train)
    y_pred_val = model.predict(X_val_top)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Validation RMSE using top features: {val_rmse:.4f}")

    return top_features 



def correlation_feature_selection(features, pred_feature, df, method='pearson', top_n=5):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    correlation_matrix = train_df[features + [pred_feature]].corr(method=method)
    target_corr = correlation_matrix[pred_feature].drop(pred_feature)

    #print(f"Correlation with Target ({method.title()} Method):")
    #print(target_corr.sort_values(ascending=False))

    selected_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()
    print(f"\nSelected features based on {method.title()} correlation: {selected_features}")

    return selected_features


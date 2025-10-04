import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('alloy_oxidation_886_202406.csv')

# Drop 'source' column
if 'source' in df.columns:
    df = df.drop('source', axis=1)

# Check shape of dataset
print(f"Dataset shape: {df.shape}")

# Encode 'Alloy formula' string column
le = LabelEncoder()
df['Alloy formula'] = le.fit_transform(df['Alloy formula'])

# Separate features and target
X = df.drop('Temperature (C)', axis=1)
y = df['Temperature (C)']

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Function to evaluate model performance with cross-validation using R-squared
def evaluate_model(X, y, features, model):
    if len(features) > 0:
        X_selected = X[features]
        scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
        return np.mean(scores)
    else:
        return -float('inf')  # Return negative infinity for empty feature sets

# 1. XGBoost feature importance
def xgboost_feature_selection(X, y, k=5):
    model = XGBRegressor(random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    feature_importance_dict = dict(zip(X.columns, feature_importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, importance in sorted_features[:k]]
    return top_features, sorted_features[0][0]  # Return top features and the highest ranked feature

# 2. Random Forest feature importance
def random_forest_feature_selection(X, y, k=5):
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    feature_importance_dict = dict(zip(X.columns, feature_importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, importance in sorted_features[:k]]
    return top_features, sorted_features[0][0]  # Return top features and the highest ranked feature

# 3. Feature Agglomeration with weighted correlation and variance
def feature_agglomeration_selection(X, y, k=5):
    # Apply feature agglomeration
    n_clusters = max(2, min(X.shape[1] - k, X.shape[1] - 1))  # Ensure valid number of clusters
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    
    # Calculate feature scores across all features using weighted correlation and variance
    feature_scores = {}
    
    # Calculate variances for normalization
    variances = X.var()
    max_variance = variances.max()
    normalized_variances = variances / max_variance if max_variance > 0 else variances
    
    # Calculate correlation-based scores
    for i, feature in enumerate(X.columns):
        # Get correlation component (lower correlation with other features is better)
        correlations = []
        for j, other_feature in enumerate(X.columns):
            if i != j:
                corr = abs(X[feature].corr(X[other_feature]))
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0
        correlation_score = 1 - avg_correlation  # Lower correlation is better
        
        # Get variance component
        variance_score = normalized_variances[feature]
        
        # Combined score with weights: 0.2 for correlation, 0.8 for variance
        feature_scores[feature] = 0.4 * correlation_score + 0.6 * variance_score
    
    # Sort features by their scores and select top k
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, score in sorted_features[:k]]
    return top_features, sorted_features[0][0]  # Return top features and the highest ranked feature

# 4. Highly Variable Gene Selection
def hvgs_feature_selection(X, y, k=5):
    # Calculate variance for each feature
    variances = X.var().sort_values(ascending=False)
    top_features = list(variances.index[:k])
    highest_feature = variances.index[0]  # Highest variance feature
    return top_features, highest_feature

# 5. Spearman correlation
def spearman_correlation_selection(X, y, k=5):
    corr_dict = {}
    for column in X.columns:
        corr_dict[column] = abs(X[column].corr(y, method='spearman'))
    
    sorted_features = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, corr in sorted_features[:k]]
    return top_features, sorted_features[0][0]  # Return top features and the highest ranked feature

# Apply all feature selection algorithms for top 5 features on full dataset
xgb_top5, xgb_highest = xgboost_feature_selection(X, y, k=5)
rf_top5, rf_highest = random_forest_feature_selection(X, y, k=5)
fa_top5, fa_highest = feature_agglomeration_selection(X, y, k=5)
hvgs_top5, hvgs_highest = hvgs_feature_selection(X, y, k=5)
spearman_top5, spearman_highest = spearman_correlation_selection(X, y, k=5)

# Evaluate models with cross-validation
xgb_model = XGBRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)

xgb_top5_score = evaluate_model(X, y, xgb_top5, xgb_model)
rf_top5_score = evaluate_model(X, y, rf_top5, rf_model)
fa_top5_score = evaluate_model(X, y, fa_top5, rf_model)  # Using RF for cross-validation only
hvgs_top5_score = evaluate_model(X, y, hvgs_top5, rf_model)  # Using RF for cross-validation only
spearman_top5_score = evaluate_model(X, y, spearman_top5, rf_model)  # Using RF for cross-validation only

print("\n--- Top 5 Features and Cross-Validation R² ---")
print(f"XGBoost: {xgb_top5}, R²: {xgb_top5_score:.4f}")
print(f"Random Forest: {rf_top5}, R²: {rf_top5_score:.4f}")
print(f"Feature Agglomeration: {fa_top5}, R²: {fa_top5_score:.4f}")
print(f"HVGS: {hvgs_top5}, R²: {hvgs_top5_score:.4f}")
print(f"Spearman: {spearman_top5}, R²: {spearman_top5_score:.4f}")

# Create reduced datasets by removing the highest feature for each algorithm
print("\n--- Highest Feature per Algorithm ---")
print(f"XGBoost highest feature: {xgb_highest}")
print(f"Random Forest highest feature: {rf_highest}")
print(f"Feature Agglomeration highest feature: {fa_highest}")
print(f"HVGS highest feature: {hvgs_highest}")
print(f"Spearman highest feature: {spearman_highest}")

# Create reduced datasets for each algorithm
X_reduced_xgb = X.drop(xgb_highest, axis=1)
X_reduced_rf = X.drop(rf_highest, axis=1)
X_reduced_fa = X.drop(fa_highest, axis=1)
X_reduced_hvgs = X.drop(hvgs_highest, axis=1)
X_reduced_spearman = X.drop(spearman_highest, axis=1)

# Apply all feature selection algorithms for top 4 features on each reduced dataset
xgb_top4, _ = xgboost_feature_selection(X_reduced_xgb, y, k=4)
rf_top4, _ = random_forest_feature_selection(X_reduced_rf, y, k=4)
fa_top4, _ = feature_agglomeration_selection(X_reduced_fa, y, k=4)
hvgs_top4, _ = hvgs_feature_selection(X_reduced_hvgs, y, k=4)
spearman_top4, _ = spearman_correlation_selection(X_reduced_spearman, y, k=4)

# Evaluate models with cross-validation on reduced dataset
xgb_top4_score = evaluate_model(X_reduced_xgb, y, xgb_top4, xgb_model)
rf_top4_score = evaluate_model(X_reduced_rf, y, rf_top4, rf_model)
fa_top4_score = evaluate_model(X_reduced_fa, y, fa_top4, rf_model)
hvgs_top4_score = evaluate_model(X_reduced_hvgs, y, hvgs_top4, rf_model)
spearman_top4_score = evaluate_model(X_reduced_spearman, y, spearman_top4, rf_model)

print("\n--- Top 4 Features from Reduced Dataset and Cross-Validation R² ---")
print(f"XGBoost: {xgb_top4}, R²: {xgb_top4_score:.4f}")
print(f"Random Forest: {rf_top4}, R²: {rf_top4_score:.4f}")
print(f"Feature Agglomeration: {fa_top4}, R²: {fa_top4_score:.4f}")
print(f"HVGS: {hvgs_top4}, R²: {hvgs_top4_score:.4f}")
print(f"Spearman: {spearman_top4}, R²: {spearman_top4_score:.4f}")

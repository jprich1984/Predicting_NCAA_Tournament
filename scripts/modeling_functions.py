import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import xgboost as xgb
import torch
import torch_geometric.data
import torch_geometric.nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import itertools
import csv
import warnings
from sklearn.neighbors import NearestNeighbors
import torch_geometric
import torch_geometric.data
import pandas as pd
import os


def ensure_categorical_columns(data, expected_columns):
    """
    Ensures that the given DataFrame contains all the expected categorical columns.
    If a column is missing, it is created and filled with False (0).

    Args:
        data (pd.DataFrame): The DataFrame to check and modify.
        expected_columns (list): List of expected column names.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """

    for col in expected_columns:
        if col not in data.columns:
            data[col] = False  # Or 0, depending on your data type
    return data

def load_xgb_model_and_predict_with_gnn(data,
                                        model_path="models/xgb_model.joblib",
                                        gnn_model_path="models/gnn_model.pth",
                                        gnn_features_path="data/processed/gnn_features.npy",
                                        k=4,
                                        out_channels=32):
    """
    Loads an XGBoost classifier, GNN model, and gnn_features from disk, generates GNN embeddings,
    and predicts on the given data.
    """

    try:
        # Get the root directory of the project
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct absolute paths
        model_path = os.path.join(root_dir, model_path)
        gnn_model_path = os.path.join(root_dir, gnn_model_path)
        gnn_features_path = os.path.join(root_dir, gnn_features_path)

        # Load gnn_features from .npy file
        gnn_features = np.load(gnn_features_path, allow_pickle=True).tolist()
        # Ensure all gnn features are present in the dataframe
        data = ensure_categorical_columns(data, gnn_features)
        # Convert boolean features to float
        for col in gnn_features:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype(float)

        # Load XGBoost Model
        xgb_model = joblib.load(model_path)
        xgb_features = xgb_model.feature_names_in_.tolist()

        # Load GNN model
        gnn_model = GraphSAGEModel(in_channels=len(gnn_features), hidden_channels=256, out_channels=out_channels, dropout_rate=0.5)
        gnn_model.load_state_dict(torch.load(gnn_model_path))
        gnn_model.eval()

        # Create graph and embeddings
        edge_index = create_graph(data, gnn_features, k=k)
        graph_data = torch_geometric.data.Data(x=torch.tensor(data[gnn_features].values, dtype=torch.float), edge_index=edge_index)
        gnn_embeddings = gnn_model(graph_data.x, graph_data.edge_index).detach().numpy()
        embedding_df = pd.DataFrame(gnn_embeddings, columns=[f'gnn_emb_{i}' for i in range(out_channels)])

        # Augment data with GNN embeddings
        data_augmented = pd.concat([data.reset_index(drop=True), embedding_df], axis=1)

        # Predict probabilities
        X_xgb = data_augmented[xgb_features]
        predicted_probabilities = xgb_model.predict_proba(X_xgb)[:, 1]
        data['Pred'] = 1 - predicted_probabilities

        return data

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def manual_grid_search_xgb_brier(data, features, target_col='efs', output_csv='data/processed/grid_search_brier.csv', mode='a', test_seasons=[[2022, 2023, 2024], [2017, 2018, 2019], [2014, 2015, 2016], [2010, 2011, 2012], [2012, 2013, 2014]]):
    """Performs manual grid search for XGBoost classifier with Brier score and custom test seasons."""

    X = data[features]
    y = data[target_col]

    param_grid = {
        'max_depth': [2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'n_estimators': [200, 300],
        'gamma': [0, 0.1, 0.2],
        'learning_rate': [0.01, 0.1, 0.3]
    }

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    param_combinations = list(itertools.product(*param_values))

    results = []

    for params in param_combinations:
        param_dict = dict(zip(param_keys, params))
        fold_brier_scores = []

        for seasons in test_seasons:
            train_indices = data[~data['Season'].isin(seasons)].index
            test_indices = data[data['Season'].isin(seasons)].index

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

            xgb_model = xgb.XGBClassifier(random_state=42, **param_dict)
            xgb_model.fit(X_train, y_train)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (loss)
            brier = brier_score_loss(y_test, y_pred_proba)
            fold_brier_scores.append(brier)

        average_brier = np.mean(fold_brier_scores)
        results.append((param_dict, average_brier))
        print(params)
        print(average_brier)

    best_params, best_brier = min(results, key=lambda x: x[1])  # Min for Brier score (lower is better)

    print("Best Parameters:", best_params)
    print("Best Brier Score:", best_brier)

    # Write results to CSV
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv_path = os.path.join(root_dir, output_csv)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, mode, newline='') as csvfile:
        fieldnames = param_keys + ['brier_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for params, brier in results:
            row = params.copy()
            row['brier_score'] = brier
            writer.writerow(row)

    best_xgb_model = xgb.XGBClassifier(random_state=42, **best_params)
    train_indices = data[~data['Season'].isin(list(itertools.chain.from_iterable(test_seasons)))].index
    best_xgb_model.fit(X.iloc[train_indices], y.iloc[train_indices])

    return best_xgb_model

def create_graph(df, features, k=4):
    """Creates a graph using K-Nearest Neighbors."""

    X = df[features].values
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.SAGEConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x
def fit_extra_trees_classifier(train, test, features, n_estimators=100, max_depth=None, min_samples_split=2, print_details=True):
    """Fits an Extra Trees Classifier."""

    X_train = train[features]
    y_train = train['efs']
    X_test = test[features]
    y_test = test['efs']

    et_model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    et_model.fit(X_train, y_train)

    y_pred = et_model.predict(X_test)
    y_prob = et_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")

    c_index = concordance_index(test['efs_time'], -y_prob, test['efs'])
    print(f"Concordance Index (using predicted probabilities): {c_index:.3f}")

    if print_details:
        # Extra Trees provide feature importances
        feature_importances = pd.Series(et_model.feature_importances_, index=features)
        print("Feature Importances:")
        print(feature_importances.sort_values(ascending=False))

    test['predicted_probability'] = y_prob

    return et_model, test, accuracy,y_prob
def fit_svm_classifier(train, test, features, C=1, kernel='rbf', gamma=.01, print_details=True):
    """Fits an SVM Classifier with scaling."""

    X_train = train[features].copy()  # Create copies to avoid modifying original data
    y_train = train['efs']
    X_test = test[features].copy()
    y_test = test['efs']

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,  # Enable probability estimates
        random_state=42
    )

    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)
    y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")

    c_index = concordance_index(test['efs_time'], -y_prob, test['efs'])
    print(f"Concordance Index (using predicted probabilities): {c_index:.3f}")

    if print_details:
        # SVMs don't have feature importances like tree-based models
        print("SVMs do not directly provide feature importances.")

    test['predicted_probability'] = y_prob

    return svm_model, test, accuracy,y_prob

def plot_top_bottom_feature_importance(feature_importances, top_n=10, bottom_n=12):
    """
    Plots the top and bottom N feature importances from a dictionary.

    Args:
        feature_importances (dict): Dictionary of feature names and their importance scores.
        top_n (int): Number of top features to plot.
        bottom_n (int): Number of bottom features to plot.
    """

    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_importances[:top_n]
    bottom_features = sorted_importances[-bottom_n:]

    # Plotting Top Features
    plt.figure(figsize=(10, 6))
    plt.bar([f[0] for f in top_features], [f[1] for f in top_features])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    # Plotting Bottom Features
    plt.figure(figsize=(10, 6))
    plt.bar([f[0] for f in bottom_features], [f[1] for f in bottom_features])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Bottom {bottom_n} Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
def add_kmeans_clusters_train_test(train_data, test_data, features, n_clusters=5, random_state=42):
    """
    Adds k-means cluster labels as a new feature to both train and test DataFrames,
    fitting on train and transforming on test.

    Args:
        train_data (pd.DataFrame): The training DataFrame containing team statistics.
        test_data (pd.DataFrame): The testing DataFrame containing team statistics.
        features (list): List of feature column names to use for clustering.
        n_clusters (int): Number of clusters to create.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (train_data, test_data) with added cluster labels.
    """

    X_train = train_data[features].copy()
    X_test = test_data[features].copy()

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply k-means clustering on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)

    # Predict clusters for test data using the fitted model
    test_clusters = kmeans.predict(X_test_scaled)

    # Add cluster labels as a new column to both train and test sets
    train_data['kmeans_cluster'] = train_clusters
    test_data['kmeans_cluster'] = test_clusters

    return train_data, test_data


def train_and_save_xgb_model(data, xgb_features, xgb_model_path="models/xgb_model.joblib", xgb_params=None, random_state=42):
    """
    Trains an XGBoost classifier and saves it to disk.

    Args:
        data (pd.DataFrame): The DataFrame containing team statistics.
        xgb_features (list): List of feature column names to use for XGBoost classifier.
        xgb_model_path (str): Path to save the XGBoost model.
        xgb_params (dict): Dictionary of XGBoost parameters. If None, default parameters are used.
        random_state (int): Random state for reproducibility.
    """

    X_xgb = data[xgb_features]
    y_xgb = data['efs']

    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.01,
            'random_state': random_state
        }

    xgb_classifier = xgb.XGBClassifier(**xgb_params)
    xgb_classifier.fit(X_xgb, y_xgb)

    # Save Model
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xgb_model_path = os.path.join(root_dir, xgb_model_path)
    os.makedirs(os.path.dirname(xgb_model_path), exist_ok=True)
    joblib.dump(xgb_classifier, xgb_model_path)

    print(f"XGBoost model saved to: {xgb_model_path}")

def load_xgb_model_and_predict(data, xgb_model_path="models/xgb_model.joblib"):
    """
    Loads an XGBoost classifier from disk and predicts on the given data.

    Args:
        data (pd.DataFrame): The DataFrame containing team statistics.
        xgb_model_path (str): Path to the XGBoost model.

    Returns:
        pd.DataFrame: The DataFrame with added 'predicted_probability' column.
    """

    try:
        # Construct absolute path
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xgb_model_path = os.path.join(root_dir, xgb_model_path)

        # Load Model
        xgb_model = joblib.load(xgb_model_path)

        # Prepare data for XGBoost
        xgb_features = xgb_model.feature_names_in_.tolist()
        X_xgb = data[xgb_features]

        # Predict probabilities
        predicted_probabilities = xgb_model.predict_proba(X_xgb)[:, 1]
        data['predicted_probability'] = predicted_probabilities

        return data

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def permutation_feature_importance_classifier(classifier, test, features, n_repeats=10, metric='accuracy'):
    """
    Calculates permutation feature importance for a classification model.

    Args:
        classifier: Fitted classification model (e.g., XGBClassifier).
        test (pd.DataFrame): Test DataFrame.
        features (list): List of feature column names.
        n_repeats (int): Number of times to permute each feature.
        metric (str): 'accuracy' for accuracy-based importance.

    Returns:
        dict: Dictionary of feature importances (change in accuracy).
    """

    X_test = test[features].copy()
    y_test = test['efs']  

    base_accuracy = accuracy_score(y_test, classifier.predict(X_test))

    feature_importances = {}

    for feature in features:
        accuracy_changes = []
        for _ in range(n_repeats):
            # Permute the feature
            X_test_permuted = X_test.copy()
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])

            # Calculate the new accuracy
            permuted_accuracy = accuracy_score(y_test, classifier.predict(X_test_permuted))
            accuracy_changes.append(base_accuracy - permuted_accuracy)

        feature_importances[feature] = np.mean(accuracy_changes)

    return feature_importances

def cross_validated_permutation_importance_seasons(model, data, features, test_seasons_list, target_col='efs', n_repeats=5):
    """
    Cross-validates permutation feature importance using different test seasons.

    Args:
        model: A classifier object.
        data (pd.DataFrame): The entire dataset.
        features (list): List of feature column names.
        test_seasons_list (list of lists): List of lists, each containing test seasons for a split.
        target_col (str): The name of the target column.
        n_repeats (int): Number of times to permute each feature in each split.

    Returns:
        dict: Dictionary of average feature importances.
    """

    feature_importances = {feature: [] for feature in features}

    for test_seasons in test_seasons_list:
        train_data, test_data = preprocess_time_split(data, test_seasons)

        X_train = train_data[features]
        y_train = train_data[target_col]
        X_test = test_data[features]
        y_test = test_data[target_col]

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Calculate permutation feature importance for the test split
        fold_importances = permutation_feature_importance_classifier(model, test_data, features, n_repeats=n_repeats)

        # Store the split importances
        for feature, importance in fold_importances.items():
            feature_importances[feature].append(importance)

    # Calculate the average importances
    average_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}

    return average_importances

def brier_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]  # Probability of class 1
    return -brier_score_loss(y, y_pred_proba)  # Negative Brier score

def fit_xgboost_classifier(train, test, features, learning_rate=0.3, subsample=0.8, n_estimators=300,gamma=0.1, max_depth=4, colsample_bytree=1.0, print_details=True):
    """Fits an XGBoost Classifier."""

    X_train = train[features]
    y_train = train['efs']
    X_test = test[features]
    y_test = test['efs']

    xgb_model = xgb.XGBClassifier(  # Use xgb.XGBClassifier
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")

    c_index = concordance_index(test['efs_time'], -y_prob, test['efs'])
    print(f"Concordance Index (using predicted probabilities): {c_index:.3f}")

    if print_details:
        # Print feature importances
        for feature, importance in zip(features, xgb_model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

    test['predicted_probability'] = y_prob

    return xgb_model, test, accuracy
def permutation_feature_importance(gb_model, test, features, n_repeats=10,metric='accuracy'):
    """
    Calculates permutation feature importance using both concordance index and accuracy.

    Args:
        gb_model: Fitted Gradient Boosting Survival Analysis model.
        test (pd.DataFrame): Test DataFrame.
        features (list): List of feature column names.
        n_repeats (int): Number of times to permute each feature.

    Returns:
        dict: Dictionary of feature importances (change in concordance index and accuracy).
    """

    X_test = test[features].copy()
    y_test_survival = test[['efs', 'efs_time']].to_records(index=False)
    y_test_survival = np.array([(bool(e), t) for e, t in y_test_survival], dtype=[('efs', bool), ('efs_time', float)])

    
    base_c_index = gb_model.score(X_test, y_test_survival)
    base_accuracy = evaluate_accuracy(test.copy())

    feature_importances = {}

    for feature in features:
        c_index_changes = []
        accuracy_changes = []
        for _ in range(n_repeats):
            # Permute the feature
            X_test_permuted = X_test.copy()
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])

            # Calculate the new concordance index
            permuted_c_index = gb_model.score(X_test_permuted, y_test_survival)
            c_index_changes.append(base_c_index - permuted_c_index)

            # Calculate the new accuracy
            test_permuted = test.copy()
            test_permuted['risk_scores'] = gb_model.predict(X_test_permuted)
            permuted_accuracy = evaluate_accuracy(test_permuted)
            accuracy_changes.append(base_accuracy - permuted_accuracy)

        if metric=='accuracy':
        # Store the average change in concordance index and accuracy
            feature_importances[feature] = np.mean(c_index_changes)
                
        else:
            feature_importances[feature] =  np.mean(accuracy_changes)

    return feature_importances

def evaluate_accuracy(test_df):
    """
    Evaluates the accuracy of predictions on the test DataFrame.

    Args:
        test_df (pd.DataFrame): DataFrame containing test data with risk scores and 'efs' column.

    Returns:
        float: Accuracy score.
    """

    winning_teams = test_df[test_df['efs'] == 0]
    losing_teams = test_df[test_df['efs'] == 1]

    games_winners = winning_teams[['TeamID', 'Season', 'Game_ID', 'DayNum', 'Game_Score', 'risk_score']].rename(
        columns={'TeamID': 'WTeamID', 'Game_Score': 'WGame_Score', 'risk_score': 'Wrisk_score'}
    )

    games_losers = losing_teams[['TeamID', 'Season', 'Game_ID', 'DayNum', 'Game_Score', 'risk_score']].rename(
        columns={'TeamID': 'LTeamID', 'Game_Score': 'LGame_Score', 'risk_score': 'Lrisk_score'}
    )

    results = pd.merge(games_winners, games_losers, on=['DayNum', 'Season', 'Game_ID'])

    results['pred_classification'] = (results['Wrisk_score'] < results['Lrisk_score']).astype(int)
    results['y_true'] = 1

    accuracy = accuracy_score(results['y_true'], results['pred_classification'])
    w_prob=sigmoid(results['Wrisk_score'])
    l_prob=sigmoid(results['Lrisk_score'])
    total_prob=w_prob+l_prob
    results['Prob']=w_prob/total_prob
    brier=brier_score_loss(results['y_true'], results['Prob']) 
    return accuracy,brier

def sigmoid(risk_score):
    """Converts a risk score to a probability using the sigmoid function."""
    probability = 1 / (1 + np.exp(risk_score))
    return probability
def risk_to_probability(test):
    test['risk_score']=test['risk_score'].apply(sigmoid)
    total_risk_df=test.groupby(['Game_ID'])['risk_score'].sum().reset_index().rename(columns={'risk_score':'total_risk'})
    joined_risk=pd.merge(test,total_risk_df,on='Game_ID')
    joined_risk['probability']=joined_risk['risk_score']/joined_risk['total_risk']

    return joined_risk.drop(columns=['total_risk'])
def preprocess_train_test_split(df, test_size=0.20, random_state=42, stratify='efs'):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify])
    return train, test
def preprocess_time_split(df, test_seasons=[2024]):
    train = df[~df['Season'].isin(  test_seasons)]
    test = df[df['Season'].isin(test_seasons)]
    return train, test
def fit_xg_survival_model(train, test, features, subsample=0.8, max_depth=4, learning_rate=0.3, n_estimators=300,show_importance=True):
    """
    Fits a Gradient Boosting Survival Analysis model.

    Args:
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame): Test DataFrame.
        features (list): List of feature column names.
        filename (str): Filename for saving the model.
        subsample (float): Subsample ratio for gradient boosting.
        max_depth (int): Maximum depth of the trees.
        learning_rate (float): Learning rate for gradient boosting.
        n_estimators (int): Number of boosting iterations.

    Returns:
        tuple: (fitted model, test DataFrame with risk scores)
    """

    # Prepare feature matrices
    X_train = train[features]
    X_test = test[features]

    # Prepare target variables (survival format)
    y_train = train[['efs', 'efs_time']].to_records(index=False)
    y_train = np.array([(bool(e), t) for e, t in y_train], dtype=[('efs', bool), ('efs_time', float)])

    y_test = test[['efs', 'efs_time']].to_records(index=False)
    y_test = np.array([(bool(e), t) for e, t in y_test], dtype=[('efs', bool), ('efs_time', float)])

    # Initialize and fit the model
    gb_model = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate the model
    c_index = gb_model.score(X_test, y_test)
    print(f"Concordance Index: {c_index:.3f}")

    # Predict risk scores
    risk_scores = gb_model.predict(X_test)

    if show_importance:
    # Print feature importances
        for feature, importance in zip(features, gb_model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

    # Add risk scores to the test DataFrame
    test['risk_score'] = risk_scores

    return gb_model, test,risk_scores

def train_and_save_models(data, kmeans_features, xgb_features, kmeans_clusters=5, random_state=42,
                            xgb_params=None, kmeans_model_path="models/kmeans_model.joblib",
                            xgb_model_path="models/xgb_model.joblib"):
    """
    Trains a KMeans clustering model and an XGBoost classifier, saves them to disk.

    Args:
        data (pd.DataFrame): The DataFrame containing team statistics.
        kmeans_features (list): List of feature column names to use for KMeans clustering.
        xgb_features (list): List of feature column names to use for XGBoost classifier.
        kmeans_clusters (int): Number of clusters for KMeans.
        random_state (int): Random state for reproducibility.
        xgb_params (dict): Dictionary of XGBoost parameters. If None, default parameters are used.
        kmeans_model_path (str): Path to save the KMeans model.
        xgb_model_path (str): Path to save the XGBoost model.
    """

    # KMeans Clustering
    X_kmeans = data[kmeans_features].copy()
    scaler_kmeans = StandardScaler()
    X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=random_state, n_init=10)
    data['kmeans_cluster'] = kmeans.fit_predict(X_kmeans_scaled)
    xgb_features = xgb_features + ['kmeans_cluster']

    # XGBoost Classifier
    X_xgb = data[xgb_features]
    y_xgb = data['efs']

    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.1,
            'random_state': random_state
        }

    xgb_classifier = xgb.XGBClassifier(**xgb_params)
    xgb_classifier.fit(X_xgb, y_xgb)

    # Save Models
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    kmeans_model_path = os.path.join(root_dir, kmeans_model_path)
    xgb_model_path = os.path.join(root_dir, xgb_model_path)
    scaler_kmeans_path = os.path.join(root_dir, 'models', 'scaler_kmeans.joblib')

    os.makedirs(os.path.dirname(kmeans_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(xgb_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_kmeans_path), exist_ok=True)

    joblib.dump(kmeans, kmeans_model_path)
    joblib.dump(scaler_kmeans, scaler_kmeans_path)
    joblib.dump(xgb_classifier, xgb_model_path)

    print(f"KMeans model saved to: {kmeans_model_path}")
    print(f"XGBoost model saved to: {xgb_model_path}")
    print(f"Kmeans Scaler saved to: {scaler_kmeans_path}")
    
def augment_data_with_gnn_embeddings(training_data, test_seasons, gnn_features, k=4, hidden_channels=128, out_channels=32, dropout_rate=0.5, weight_decay=5e-4, epochs=1200):
    """Augments data with GNN embeddings."""

    train, test = preprocess_time_split(training_data, test_seasons=test_seasons)
    train_df, val_df = preprocess_time_split(train, test_seasons=[2019, 2021])

    for col in gnn_features:
        if train_df[col].dtype == 'bool':
            train_df[col] = train_df[col].astype(float)
        if val_df[col].dtype == 'bool':
            val_df[col] = val_df[col].astype(float)
        if test[col].dtype == 'bool':
            test[col] = test[col].astype(float)

    train_edge_index = create_graph(train_df, gnn_features, k=k)
    val_edge_index = create_graph(val_df, gnn_features, k=k)

    train_data = torch_geometric.data.Data(x=torch.tensor(train_df[gnn_features].values, dtype=torch.float), edge_index=train_edge_index)
    val_data = torch_geometric.data.Data(x=torch.tensor(val_df[gnn_features].values, dtype=torch.float), edge_index=val_edge_index)

    in_channels = len(gnn_features)

    model = GraphSAGEModel(in_channels, hidden_channels, out_channels, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = F.mse_loss(out, train_data.x[:, :out_channels])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index)
            val_loss = F.mse_loss(val_out, val_data.x[:, :out_channels])
            if epoch % 1200 == 0:
                print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
            if val_loss < 2:
                print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
        model.train()

    model.eval()
    train_embeddings = model(train_data.x, train_data.edge_index).detach().numpy()
    val_embeddings = model(val_data.x, val_data.edge_index).detach().numpy()

    train_embedding_df = pd.DataFrame(train_embeddings, columns=[f'gnn_emb_{i}' for i in range(out_channels)])
    val_embedding_df = pd.DataFrame(val_embeddings, columns=[f'gnn_emb_{i}' for i in range(out_channels)])

    training_data_augmented = pd.concat([train_df.reset_index(drop=True), train_embedding_df], axis=1)
    val_dummied_augmented = pd.concat([val_df.reset_index(drop=True), val_embedding_df], axis=1)

    test_edge_index = create_graph(test, gnn_features, k=k)
    test_data = torch_geometric.data.Data(x=torch.tensor(test[gnn_features].values, dtype=torch.float), edge_index=test_edge_index)
    test_embeddings = model(test_data.x, test_data.edge_index).detach().numpy()
    test_embedding_df = pd.DataFrame(test_embeddings, columns=[f'gnn_emb_{i}' for i in range(out_channels)])

    test_augmented = pd.concat([test.reset_index(drop=True), test_embedding_df], axis=1)

    data_augmented = pd.concat([training_data_augmented, val_dummied_augmented])
    return data_augmented,test_augmented

def manual_grid_search_svm_brier(data, features, target_col='efs', mode='a', output_csv='data/processed/svm_grid_search_brier.csv',
                                    test_seasons=[[2022, 2023, 2024], [2017, 2018, 2019], [2014, 2015, 2016], [2010, 2011, 2012], [2012, 2013, 2014]]):
    """Performs manual grid search for SVM classifier with Brier score and custom test seasons."""

    X = data[features]
    y = data[target_col]

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1, 0.01],  # Added 0.01 to gamma
        'probability': [True]
    }

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())

    param_combinations = list(itertools.product(*param_values))

    results = []

    for params in param_combinations:
        param_dict = dict(zip(param_keys, params))
        fold_brier_scores = []

        for seasons in test_seasons:
            train_indices = data[~data['Season'].isin(seasons)].index
            test_indices = data[data['Season'].isin(seasons)].index

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            svm_model = SVC(random_state=42, **param_dict)
            svm_model.fit(X_train_scaled, y_train)
            y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
            brier = brier_score_loss(y_test, y_pred_proba)
            fold_brier_scores.append(brier)

        average_brier = np.mean(fold_brier_scores)
        results.append((param_dict, average_brier))
        print(params)
        print(average_brier)

    best_params, best_brier = min(results, key=lambda x: x[1])

    print("Best Parameters:", best_params)
    print("Best Brier Score:", best_brier)

    # Construct absolute path for csv file
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv_path = os.path.join(root_dir, output_csv)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, mode, newline='') as csvfile:
        fieldnames = param_keys + ['brier_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for params, brier in results:
            row = params.copy()
            row['brier_score'] = brier
            writer.writerow(row)

    best_svm_model = SVC(random_state=42, **best_params)

    return best_svm_model
def train_and_save_svm_models_women(data, kmeans_features, svm_features, kmeans_clusters=5, random_state=42,
                                        svm_params=None, kmeans_model_path="models/kmeans_model_women.joblib",
                                        svm_model_path="models/svm_model_women.joblib",
                                        kmeans_features_path="data/processed/kmeans_features_women.joblib",
                                        svm_features_path="data/processed/svm_features_women.joblib",
                                        svm_scaler_path="models/svm_scaler_women.joblib"):
    """
    Trains a KMeans clustering model, scales SVM features, and trains an SVM classifier, saves them to disk.

    Args:
        data (pd.DataFrame): The DataFrame containing team statistics.
        kmeans_features (list): List of feature column names to use for KMeans clustering.
        svm_features (list): List of feature column names to use for SVM classifier.
        kmeans_clusters (int): Number of clusters for KMeans.
        random_state (int): Random state for reproducibility.
        svm_params (dict): Dictionary of SVM parameters. If None, default parameters are used.
        kmeans_model_path (str): Path to save the KMeans model.
        svm_model_path (str): Path to save the SVM model.
        kmeans_features_path (str): Path to save kmeans_features list.
        svm_features_path (str): Path to save svm_features list.
        svm_scaler_path (str) : Path to save svm scaler.
    """

    # KMeans Clustering
    X_kmeans = data[kmeans_features].copy()
    scaler_kmeans = StandardScaler()
    X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=random_state, n_init=10)
    data['kmeans_cluster'] = kmeans.fit_predict(X_kmeans_scaled)
    svm_features = svm_features + ['kmeans_cluster']

    # SVM Classifier with Scaled Features
    X_svm = data[svm_features].copy()
    scaler_svm = StandardScaler()
    X_svm_scaled = scaler_svm.fit_transform(X_svm)
    y_svm = data['efs']

    if svm_params is None:
        svm_params = {
            'C': 1,
            'gamma': 0.01,
            'kernel': 'rbf',
            'random_state': random_state,
            'probability': True
        }

    svm_classifier = SVC(**svm_params)
    svm_classifier.fit(X_svm_scaled, y_svm)

    # Save Models and Scalers
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    kmeans_model_path = os.path.join(root_dir, kmeans_model_path)
    svm_model_path = os.path.join(root_dir, svm_model_path)
    kmeans_features_path = os.path.join(root_dir, kmeans_features_path)
    svm_features_path = os.path.join(root_dir, svm_features_path)
    svm_scaler_path = os.path.join(root_dir, svm_scaler_path)
    scaler_kmeans_path = os.path.join(root_dir, 'data', 'processed', 'scaler_kmeans_women.joblib')

    os.makedirs(os.path.dirname(kmeans_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(svm_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(kmeans_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(svm_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(svm_scaler_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_kmeans_path), exist_ok=True)

    joblib.dump(kmeans, kmeans_model_path)
    joblib.dump(scaler_kmeans, scaler_kmeans_path)
    joblib.dump(svm_classifier, svm_model_path)
    joblib.dump(kmeans_features, kmeans_features_path)
    joblib.dump(svm_features, svm_features_path)
    joblib.dump(scaler_svm, svm_scaler_path)

    print(f"KMeans model saved to: {kmeans_model_path}")
    print(f"SVM model saved to: {svm_model_path}")
    print(f"Kmeans Scaler saved to: {scaler_kmeans_path}")
    print(f"KMeans features saved to: {kmeans_features_path}")
    print(f"SVM features saved to: {svm_features_path}")
    print(f"SVM scaler saved to: {svm_scaler_path}")

def load_svm_model_and_predict_women(data, kmeans_model_path="models/kmeans_model_women.joblib",
                                        scaler_path="models/scaler_kmeans_women.joblib",
                                        svm_model_path="models/svm_model_women.joblib",
                                        kmeans_features_path="data/processed/kmeans_features_women.joblib",
                                        svm_features_path="data/processed/svm_features_women.joblib",
                                        svm_scaler_path="models/svm_scaler_women.joblib"):
    """
    Loads a KMeans clustering model, scalers, and SVM classifier from disk and predicts on the given data.

    Args:
        data (pd.DataFrame): The DataFrame containing team statistics.
        kmeans_model_path (str): Path to the KMeans model.
        scaler_path (str): Path to the StandardScaler for kmeans.
        svm_model_path (str): Path to the SVM model.
        kmeans_features_path (str): Path to the KMeans features list.
        svm_features_path (str): Path to the SVM features list.
        svm_scaler_path (str) : Path to the scaler for svm features.

    Returns:
        pd.DataFrame: The DataFrame with added 'predicted_probability' column.
    """
    try:
        # Construct absolute paths
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kmeans_model_path = os.path.join(root_dir, kmeans_model_path)
        scaler_path = os.path.join(root_dir, scaler_path)
        svm_model_path = os.path.join(root_dir, svm_model_path)
        kmeans_features_path = os.path.join(root_dir, kmeans_features_path)
        svm_features_path = os.path.join(root_dir, svm_features_path)
        svm_scaler_path = os.path.join(root_dir, svm_scaler_path)

        # Load Models and Scalers
        kmeans = joblib.load(kmeans_model_path)
        scaler_kmeans = joblib.load(scaler_path)
        svm_model = joblib.load(svm_model_path)
        kmeans_features = joblib.load(kmeans_features_path)
        svm_features = joblib.load(svm_features_path)
        scaler_svm = joblib.load(svm_scaler_path)

        # Prepare data for KMeans
        X_kmeans = data[kmeans_features].copy()
        X_kmeans_scaled = scaler_kmeans.transform(X_kmeans)
        data['kmeans_cluster'] = kmeans.predict(X_kmeans_scaled)
        svm_features = svm_features

        # Prepare data for SVM
        X_svm = data[svm_features].copy()
        X_svm_scaled = scaler_svm.transform(X_svm)

        # Predict probabilities
        predicted_probabilities = svm_model.predict_proba(X_svm_scaled)[:, 1]
        data['Pred'] = 1 - predicted_probabilities

        return data

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


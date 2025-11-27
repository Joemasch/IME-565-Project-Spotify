import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Audio features used for mood clustering
FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness"
]


def load_kaggle_dataset(path: str = "data/Kaggle.csv"):
    """
    Load the Kaggle dataset into a pandas DataFrame.

    Args:
        path (str): Path to the Kaggle CSV file. Defaults to "data/Kaggle.csv".

    Returns:
        pd.DataFrame: Loaded dataset with song metadata and audio features.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the specified path.
        ValueError: If required audio feature columns are missing from the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kaggle dataset not found at path: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file at {path}: {str(e)}")

    # Check that all required feature columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required audio feature columns: {missing_cols}")

    return df


def train_mood_models(df, feature_cols, n_clusters: int = 8):
    """
    Train the StandardScaler and KMeans models for mood clustering.

    Args:
        df (pd.DataFrame): Kaggle dataset DataFrame.
        feature_cols (list): List of audio feature column names to use for clustering.
        n_clusters (int): Number of mood clusters to create. Defaults to 8.

    Returns:
        tuple: (scaler, kmeans) - trained StandardScaler and KMeans models.

    Raises:
        ValueError: If feature_cols contains invalid column names or n_clusters is invalid.
    """
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("feature_cols must be a non-empty list")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing feature columns: {missing_cols}")

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    # Extract feature matrix
    try:
        X = df[feature_cols].values
    except Exception as e:
        raise ValueError(f"Failed to extract feature matrix: {str(e)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    return scaler, kmeans


def assign_mood_clusters(df, scaler, kmeans, feature_cols):
    """
    Assign mood cluster IDs to each song in the dataset.

    Args:
        df (pd.DataFrame): Kaggle dataset DataFrame.
        scaler: Trained StandardScaler instance.
        kmeans: Trained KMeans instance.
        feature_cols (list): List of audio feature column names.

    Returns:
        pd.DataFrame: DataFrame with added 'mood_cluster' column containing cluster IDs.

    Raises:
        ValueError: If required columns are missing or models are invalid.
    """
    if 'mood_cluster' in df.columns:
        print("Warning: 'mood_cluster' column already exists, it will be overwritten")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing feature columns: {missing_cols}")

    try:
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        cluster_labels = kmeans.predict(X_scaled)

        df_copy = df.copy()
        df_copy['mood_cluster'] = cluster_labels

        return df_copy

    except Exception as e:
        raise ValueError(f"Failed to assign mood clusters: {str(e)}")


def label_moods_by_cluster(df, feature_cols, cluster_col: str = "mood_cluster"):
    """
    Analyze cluster centroids and assign human-readable mood names to each cluster.

    Uses heuristic rules based on average audio feature values within each cluster:
    - High energy & high valence → "Upbeat / Party"
    - High acousticness & lower energy → "Chill / Acoustic"
    - High instrumentalness → "Instrumental / Focus"
    - Low valence → "Melancholy / Sad"
    - High danceability → "Dance / Energetic"
    - High speechiness → "Vocal / Speech"
    - Default fallback → "Mixed / Ambient"

    Args:
        df (pd.DataFrame): DataFrame with mood_cluster column.
        feature_cols (list): List of audio feature column names.
        cluster_col (str): Name of the cluster column. Defaults to "mood_cluster".

    Returns:
        dict: Mapping of cluster_id to mood_name.

    Raises:
        ValueError: If cluster_col doesn't exist or feature_cols are invalid.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{cluster_col}' column")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing feature columns: {missing_cols}")

    try:
        # Group by cluster and compute mean feature values
        cluster_means = df.groupby(cluster_col)[feature_cols].mean()

        mood_labels = {}

        for cluster_id, features in cluster_means.iterrows():
            energy = features['energy']
            valence = features['valence']
            acousticness = features['acousticness']
            instrumentalness = features['instrumentalness']
            danceability = features['danceability']
            speechiness = features['speechiness']

            # Heuristic mood assignment rules
            if energy > 0.7 and valence > 0.6:
                mood_name = "Upbeat / Party"
            elif acousticness > 0.6 and energy < 0.5:
                mood_name = "Chill / Acoustic"
            elif instrumentalness > 0.5:
                mood_name = "Instrumental / Focus"
            elif valence < 0.4:
                mood_name = "Melancholy / Sad"
            elif danceability > 0.7:
                mood_name = "Dance / Energetic"
            elif speechiness > 0.3:
                mood_name = "Vocal / Speech"
            else:
                mood_name = "Mixed / Ambient"

            mood_labels[cluster_id] = mood_name

        return mood_labels

    except Exception as e:
        raise ValueError(f"Failed to label moods by cluster: {str(e)}")


def build_and_save_mood_space(
    input_path: str = "data/Kaggle.csv",
    output_path: str = "data/Kaggle_with_moods.csv",
    n_clusters: int = 8,
):
    """
    Complete pipeline to build and save the mood clustering space.

    This function orchestrates the entire mood clustering process:
    1. Loads the Kaggle dataset
    2. Trains scaler and KMeans models
    3. Assigns cluster IDs to songs
    4. Labels clusters with human-readable mood names
    5. Saves the updated dataset with mood information
    6. Saves the trained models for future use

    Args:
        input_path (str): Path to input Kaggle CSV file. Defaults to "data/Kaggle.csv".
        output_path (str): Path to save the dataset with moods. Defaults to "data/Kaggle_with_moods.csv".
        n_clusters (int): Number of mood clusters to create. Defaults to 8.

    Returns:
        pd.DataFrame: Dataset with added mood_cluster and mood_name columns.

    Raises:
        Various exceptions from individual pipeline steps.
    """
    print("Loading Kaggle dataset...")
    df = load_kaggle_dataset(input_path)

    print(f"Training mood models with {n_clusters} clusters...")
    scaler, kmeans = train_mood_models(df, FEATURE_COLS, n_clusters)

    print("Assigning mood clusters to songs...")
    df_with_clusters = assign_mood_clusters(df, scaler, kmeans, FEATURE_COLS)

    print("Labeling clusters with mood names...")
    mood_labels = label_moods_by_cluster(df_with_clusters, FEATURE_COLS)

    # Add mood names to DataFrame
    df_with_clusters['mood_name'] = df_with_clusters['mood_cluster'].map(mood_labels)

    print(f"Saving updated dataset to {output_path}...")
    df_with_clusters.to_csv(output_path, index=False)

    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("Saving trained models...")
    joblib.dump(scaler, os.path.join(models_dir, "mood_scaler.pkl"))
    joblib.dump(kmeans, os.path.join(models_dir, "mood_kmeans.pkl"))

    # Save mood labels mapping
    joblib.dump(mood_labels, os.path.join(models_dir, "mood_labels.pkl"))

    print("Mood space building complete!")
    return df_with_clusters

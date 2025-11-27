# Music Taste Mood Clustering Specification

This document describes how we will create “mood clusters” using audio features from the Kaggle dataset.

The goal:
- Group songs into meaningful clusters based on their audio characteristics
- Give each cluster a human-friendly mood label
- Use these clusters to help users understand their music taste
- Support future Spotify-API integration (mapping a user’s songs into the same mood space)

## Dataset Used for Mood Clustering

We will use the Kaggle Spotify dataset stored at:

```
data/Kaggle.csv
```

Each row in this dataset represents one song.  
The dataset includes:

- Song metadata (track name, artist, album, genre)
- Spotify-style audio features:
  - danceability
  - energy
  - valence
  - acousticness
  - instrumentalness
  - liveness
  - speechiness
  - tempo
  - loudness
  - duration_ms

  ## Audio Features Used for Mood Clustering

We will use a small set of numeric audio features from the Kaggle dataset to build mood clusters.

These features describe the character of a song:

```python
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
```

These will be scaled and used to train a k-means clustering model.

## Building the Mood Space

We will create a global “mood space” by clustering the audio features.

The steps are:

1. Load the Kaggle dataset from `data/Kaggle.csv`
2. Select the `FEATURE_COLS`
3. Scale the features using `StandardScaler`
4. Fit a `KMeans` clustering model (we will start with 8 clusters)
5. Assign each song:
   - a numeric cluster ID (`mood_cluster`)
   - a human-readable label (`mood_name`)
6. Save:
   - a new CSV file: `data/Kaggle_with_moods.csv`
   - the scaler and k-means model in `models/`

This mood space will later be used to understand user taste and help users explore music by mood.

## Functions to Implement (for Cursor)

We will put the mood-clustering logic into a Python file:

`app/moods.py`

This file should contain a few clear functions.

### 1. Load the Kaggle dataset

```python
def load_kaggle_dataset(path: str = "data/Kaggle.csv"):
    """
    Load the Kaggle dataset into a pandas DataFrame.
    Check that the important audio feature columns exist.
    If the file or columns are missing, raise a clear error.
    """
```

### 2. Train the scaler and KMeans models

```python
def train_mood_models(df, feature_cols, n_clusters: int = 8):
    """
    Take the Kaggle DataFrame and the list of audio feature columns.

    Steps:
    - Extract the feature matrix df[feature_cols]
    - Scale it using StandardScaler
    - Fit a KMeans model with n_clusters
    - Return (scaler, kmeans)
    """
```

### 3. Assign mood clusters to every song

```python
def assign_mood_clusters(df, scaler, kmeans, feature_cols):
    """
    Use the trained scaler and kmeans model to assign a cluster to each song.

    Steps:
    - Transform df[feature_cols] using the scaler
    - Predict cluster IDs using kmeans
    - Add a new column df["mood_cluster"]
    - A second step will later add df["mood_name"]
    - Return the updated DataFrame
    """
```

### 4. Give each cluster a human-readable mood name

```python
def label_moods_by_cluster(df, feature_cols, cluster_col: str = "mood_cluster"):
    """
    Look at the average features inside each cluster and assign a simple mood name.

    Example rules (can be heuristic, not perfect):
    - High energy & high valence → "Upbeat / Party"
    - High acousticness & lower energy → "Chill / Acoustic"
    - High instrumentalness → "Instrumental / Focus"
    - Low valence → "Melancholy / Sad"

    This function should:
    - Group the DataFrame by cluster_col
    - Compute mean values of feature_cols per cluster
    - Decide a name for each cluster
    - Return a dict: {cluster_id: mood_name}
    """
```

### 5. Full pipeline helper: build and save the mood space

```python
def build_and_save_mood_space(
    input_path: str = "data/Kaggle.csv",
    output_path: str = "data/Kaggle_with_moods.csv",
    n_clusters: int = 8,
):
    """
    High-level helper that runs the whole pipeline:

    - Load the Kaggle dataset from input_path
    - Train the scaler and kmeans using the chosen feature columns
    - Assign mood_cluster to each song
    - Use label_moods_by_cluster to get names for each cluster
    - Add a 'mood_name' column to the DataFrame
    - Save the updated DataFrame to output_path
    - Save the scaler and kmeans models to the models/ folder
    """
```







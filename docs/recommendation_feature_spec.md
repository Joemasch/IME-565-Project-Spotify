# Recommendation Feature Specification

This document describes the plan for building a transparent, user-controlled music recommendation feature. It will guide Cursor so it can generate the code correctly.

The recommendation system will:
- Use song audio-feature data from a CSV
- Use standard machine-learning models (classification + regression)
- Generate recommendations with adjustable controls (novelty, popularity)
- Provide human-readable explanations for each recommended song
- Work entirely offline without Spotify’s API for now

## Data the recommendation system will use

The app will work with a table of songs stored in a CSV file.

Each row = one song.  
Each column = information about that song.

We expect columns like:
- `track_name`
- `artist_name`
- `danceability`
- `energy`
- `valence`
- `tempo`
- `loudness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `speechiness`
- `popularity` (0–100, can be used as a regression target)
- `like_song` (0 or 1, can be used as a classification target)

## Machine Learning Models the System Will Use

The app will support two types of prediction:

### 1. Classification
Goal: Predict whether the user will “like” a song (0 or 1).

Models we will use:
- Logistic Regression
- Random Forest Classifier

### 2. Regression
Goal: Predict a numeric score such as `popularity`.

Models we will use:
- Linear Regression
- Random Forest Regressor

## User Controls in the App

The user will be able to adjust how recommendations are generated:

### 1. Novelty Slider (0 to 1)
- 0 = show songs similar to the user’s usual taste  
- 1 = show songs that are more exploratory and different

### 2. Popularity Slider (0 to 1)
- 0 = prefer niche or less popular songs  
- 1 = prefer mainstream songs

### 3. Model Selector
The user chooses which ML model to use for prediction.

### 4. Task Selector
The user chooses between:
- Classification (predict like vs not like)
- Regression (predict popularity / score)

## Recommendation Logic (Simple Version)

The system will generate recommendations using these steps:

### 1. Compute a “user profile”
This is the average of the audio features for the songs the user likes.

Example:
```python
user_profile = liked_songs[FEATURE_COLS].mean(axis=0)
```

### 2. Measure similarity to the user profile
Each song in the dataset will get a similarity score based on its audio features.

### 3. Apply novelty adjustment
If novelty slider = 0 → prefer similar songs  
If novelty slider = 1 → prefer different songs

### 4. Apply popularity adjustment
Adjust scores based on whether the user prefers niche or popular songs.

### 5. Generate a final score
We combine:
- similarity
- novelty effect
- popularity effect
- predicted “like probability” or predicted score

### 6. Select the top N songs
The system returns the highest-scoring songs with explanations.

## App UI (Streamlit) – Simple Plan

The recommendation feature will live on its own page in the Streamlit app.

That page will include:

### Sidebar
- File uploader for the song CSV
- Task selector:
  - Classification
  - Regression
- Model selector (for the chosen task)
- Novelty slider (0 to 1)
- Popularity slider (0 to 1)

### Main Area
- A button: **"Generate Recommendations"**
- A table showing the recommended songs and their scores
- A short text explanation for each recommended song

### Extra Visuals (later)
We can also show:
- For classification: a confusion matrix and feature importance bar chart
- For regression: a residual plot and predicted vs actual scatter plot








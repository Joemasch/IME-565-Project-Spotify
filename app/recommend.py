def train_classification_models(X, y):
    """
    Train and return a dictionary of classification models.
    Keys will be model names (e.g., 'Logistic Regression', 'Random Forest Classifier').
    Values will be fitted scikit-learn model objects.
    """

def train_regression_models(X, y):
    """
    Train and return a dictionary of regression models.
    Keys will be model names (e.g., 'Linear Regression', 'Random Forest Regressor').
    Values will be fitted scikit-learn model objects.
    """

def compute_user_profile(df, feature_cols, like_col="like_song"):
    """
    Compute the average feature vector for songs the user likes.
    For now, this can just take rows where like_col == 1 and average feature_cols.
    """

def score_songs_simple(df, feature_cols, user_profile):
    """
    Placeholder function.
    For now, just compute a simple similarity score between each song and the user_profile.
    This will be replaced later with the full recommendation logic (novelty, popularity, etc.).
    """

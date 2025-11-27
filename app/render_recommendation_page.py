import streamlit as st
import pandas as pd
import numpy as np


def build_explanation(selected_row, rec_row, feature_cols):
    """
    Build a short text explanation comparing rec_row to selected_row
    using a few of the numeric audio features.
    """
    available_features = [c for c in feature_cols if c in selected_row.index and c in rec_row.index]

    if not available_features:
        return "Recommended based on overall similarity."

    diffs = {}
    for col in available_features:
        base = selected_row[col]
        rec = rec_row[col]
        diffs[col] = rec - base

    # Pick a few features with the largest absolute differences
    sorted_features = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [name for name, diff in sorted_features[:3]]

    # Build a human-readable sentence
    parts = []
    for name in top_features:
        diff = diffs[name]
        direction = "higher" if diff > 0 else "lower"
        parts.append(f"{direction} {name}")

    if not parts:
        return "Recommended based on overall similarity."

    joined = ", ".join(parts)
    return f"Compared to your selected song, this one has {joined}."


def render_recommendation_page():
    st.title("Recommendation System")
    st.write("This page will be used later to experiment with recommendations.")

    # CSV file uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Create display_name column for better song identification
        if "track_name" in df.columns:
            if "artists" in df.columns:
                df["display_name"] = df["track_name"] + " — " + df["artists"]
            else:
                df["display_name"] = df["display_name"] = df["track_name"]

        # Deduplicate the DataFrame to ensure each song appears at most once
        if "track_id" in df.columns:
            df_dedup = df.drop_duplicates(subset=["track_id"]).copy()
        elif "track_name" in df.columns and "artists" in df.columns:
            df_dedup = df.drop_duplicates(subset=["track_name", "artists"]).copy()
        else:
            df_dedup = df.drop_duplicates().copy()

        # Show preview message
        st.write("Preview of uploaded data:")

        # Display first 10 rows
        st.dataframe(df.head(10))

        # Song selector (only if track_name column exists)
        if "track_name" in df.columns:
            # Get unique display names (includes track_name and artists if available)
            song_options = df_dedup["display_name"].unique()

            # Create selectbox for song selection
            selected_display_name = st.selectbox(
                "Select a song from the dataset",
                options=song_options
            )

            # Get the selected row and display it
            if selected_display_name:
                selected_row = df_dedup[df_dedup["display_name"] == selected_display_name].iloc[0]
                st.dataframe(selected_row.to_frame().T)

                # Recommendation settings
                st.subheader("Recommendation Settings")
                num_recs = st.slider(
                    "Number of recommended songs",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                )

                novelty = st.slider(
                    "Novelty (0 = very similar, 1 = more different)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                )

                popularity_pref = st.slider(
                    "Popularity Preference (0 = niche, 1 = mainstream)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                )

                artist_diversity = st.slider(
                    "Artist diversity (0 = okay with repeats, 1 = prefer many artists)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                )

                # Similar songs section
                st.subheader("Similar songs")

                # Define possible feature columns
                feature_cols = [
                    "danceability",
                    "energy",
                    "valence",
                    "acousticness",
                    "instrumentalness",
                    "liveness",
                    "speechiness",
                    "tempo",
                    "loudness",
                ]

                # Check which features are actually present in the DataFrame
                available_features = [col for col in feature_cols if col in df.columns]

                if not available_features:
                    st.info("No audio features found for similarity calculation.")
                else:
                    # Use the selected_row as the anchor song
                    anchor_features = selected_row[available_features].values

                    # Calculate Euclidean distance for all songs
                    def calculate_distance(row):
                        song_features = row[available_features].values
                        return np.linalg.norm(song_features - anchor_features)

                    # Add distance column to a copy of the DataFrame
                    df_with_distance = df_dedup.copy()
                    df_with_distance['distance_to_selected'] = df_with_distance.apply(calculate_distance, axis=1)

                    # Filter out the selected song itself (by index)
                    similar_df = df_with_distance[
                        df_with_distance.index != selected_row.name
                    ]

                    # Sort based on novelty preference
                    if novelty <= 0.5:
                        # Familiar: small distance is better (most similar)
                        similar_df = similar_df.sort_values("distance_to_selected", ascending=True)
                    else:
                        # Exploratory: large distance is better (most different)
                        similar_df = similar_df.sort_values("distance_to_selected", ascending=False)

                    # Apply popularity preference (if popularity column exists)
                    if "popularity" in similar_df.columns:
                        if popularity_pref <= 0.5:
                            # Niche preference: lower popularity is better
                            similar_df = similar_df.sort_values("popularity", ascending=True)
                        else:
                            # Mainstream preference: higher popularity is better
                            similar_df = similar_df.sort_values("popularity", ascending=False)
                    else:
                        st.info("No popularity column found; popularity slider has no effect.")

                    # Apply artist diversity adjustment (if artists column exists and diversity > 0.5)
                    if "artists" in similar_df.columns and artist_diversity > 0.5:
                        artist_counts = similar_df["artists"].value_counts()
                        similar_df["artist_count"] = similar_df["artists"].map(artist_counts)
                        # For high artist_diversity, we want artists that appear less often near the top
                        similar_df = similar_df.sort_values("artist_count", ascending=True)

                    # Generate explanations for each recommendation
                    explanations = []
                    for _, rec_row in similar_df.iterrows():
                        explanations.append(build_explanation(selected_row, rec_row, feature_cols))
                    similar_df["explanation"] = explanations

                    # Limit to selected number of recommendations
                    similar_df = similar_df.head(num_recs)

                    # Use consistent variable name
                    similar_songs = similar_df

                    # Select columns to display (prioritize the ones mentioned)
                    display_cols = ['track_name']
                    if 'artists' in similar_songs.columns:
                        display_cols.append('artists')
                    if 'mood_name' in similar_songs.columns:
                        display_cols.append('mood_name')
                    display_cols.append('distance_to_selected')
                    display_cols.append('explanation')

                    # Display the similar songs
                    st.dataframe(similar_songs[display_cols])
        else:
            st.info("No song selector is available (track_name column not found in the dataset).")

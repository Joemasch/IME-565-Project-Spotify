import streamlit as st
import pandas as pd
import os
from moods import FEATURE_COLS


def render_taste_explorer_page():
    """
    Render the Taste Explorer page that displays mood-clustered song data.

    This function loads the Kaggle dataset with mood information and displays
    basic information about the songs and their mood clusters.
    """
    # Load the dataset with mood information
    # Use absolute path to ensure it works regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "Kaggle_with_moods.csv")
    df = pd.read_csv(data_path)

    # Display title
    st.title("Taste Explorer")

    # Show the number of songs
    st.write(f"Total songs: {len(df):,}")

    # Show the first 10 rows
    st.dataframe(df.head(10))

    # Show mood distribution bar chart
    st.subheader("Mood Distribution")
    mood_counts = df["mood_name"].value_counts()
    st.bar_chart(mood_counts)

    # Mood characteristics section
    st.subheader("Mood Characteristics")
    selected_mood = st.selectbox(
        "Select a mood to inspect",
        options=sorted(df["mood_name"].unique())
    )

    if selected_mood:
        # Filter data for selected mood and compute averages
        mood_data = df[df["mood_name"] == selected_mood]
        mood_averages = mood_data[FEATURE_COLS].mean()

        # Display the averages as a bar chart
        st.bar_chart(mood_averages)

import streamlit as st
from render_taste_explorer_page import render_taste_explorer_page
from render_recommendation_page import render_recommendation_page


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Taste Explorer", "Recommendation System"])

    if page == "Taste Explorer":
        render_taste_explorer_page()
    elif page == "Recommendation System":
        render_recommendation_page()

if __name__ == "__main__":
    main()


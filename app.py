import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="AI Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 AI Movie Recommendation System")
st.markdown("Content-Based • Genre-Based • Hybrid Recommendation")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# ===================== CLEAN DATA =====================
movies["genres"] = movies["genres"].fillna("")
movies["title"] = movies["title"].fillna("")
ratings = ratings.dropna()

# Create a combined text column (VERY IMPORTANT)
movies["text"] = movies["title"] + " " + movies["genres"]

# ===================== CONTENT-BASED MODEL =====================
@st.cache_data
def build_similarity(data):
    cv = CountVectorizer(stop_words="english")
    count_matrix = cv.fit_transform(data["text"])
    similarity = cosine_similarity(count_matrix)
    return similarity

similarity_matrix = build_similarity(movies)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# ===================== RECOMMENDER FUNCTIONS =====================
def recommend_content(title, n=5):
    try:
        if title not in indices:
            return []

        idx = indices[title]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]

        movie_indices = [i[0] for i in scores]
        return movies["title"].iloc[movie_indices].tolist()
    except:
        return []

def recommend_genre(genre, n=5):
    filtered = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    if len(filtered) == 0:
        return []
    return filtered["title"].sample(min(n, len(filtered))).tolist()

def recommend_hybrid(title, n=5):
    try:
        content_recs = recommend_content(title, n=30)
        if len(content_recs) == 0:
            return []

        avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
        merged = movies.merge(avg_ratings, on="movieId", how="left")
        merged["rating"] = merged["rating"].fillna(0)

        candidates = merged[merged["title"].isin(content_recs)]
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(by="rating", ascending=False)
        return candidates["title"].head(n).tolist()
    except:
        return []

# ===================== UI =====================
st.subheader("Choose Recommendation Type:")

rec_type = st.radio(
    "",
    ["Content-Based", "Genre-Based", "Hybrid"],
    horizontal=True
)

st.divider()

# ===================== CONTENT / HYBRID =====================
if rec_type in ["Content-Based", "Hybrid"]:
    movie_list = sorted(movies["title"].unique())
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    if st.button("Recommend"):
        if rec_type == "Content-Based":
            recommendations = recommend_content(selected_movie)
        else:
            recommendations = recommend_hybrid(selected_movie)

        st.subheader("Recommended Movies:")
        if len(recommendations) == 0:
            st.warning("No recommendations found for this movie.")
        else:
            for movie in recommendations:
                st.write("⭐", movie)

# ===================== GENRE =====================
elif rec_type == "Genre-Based":
    all_genres = set()
    for g in movies["genres"]:
        for x in g.split("|"):
            if x.strip() != "":
                all_genres.add(x.strip())

    genre = st.selectbox("Choose a genre:", sorted(all_genres))

    if st.button("Recommend"):
        recommendations = recommend_genre(genre)

        st.subheader("Recommended Movies:")
        if len(recommendations) == 0:
            st.warning("No movies found for this genre.")
        else:
            for movie in recommendations:
                st.write("⭐", movie)

# ===================== FOOTER =====================
st.divider()
st.caption("Built by Sathvikha Reddy • AI Movie Recommendation System")

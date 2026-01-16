import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="AI Movie Recommendation System", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.write("Content-Based and Genre-Based Movie Recommendation System")

# ===================== LOAD DATA =====================
@st.cache_data
def load_movies():
    movies = pd.read_csv("movies.csv")
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("")
    movies = movies.reset_index(drop=True)
    movies["text"] = movies["title"] + " " + movies["genres"]
    return movies

movies = load_movies()

# ===================== BUILD SIMILARITY =====================
@st.cache_data
def build_similarity(data):
    cv = CountVectorizer(stop_words="english")
    matrix = cv.fit_transform(data["text"])
    sim = cosine_similarity(matrix)
    return sim

similarity = build_similarity(movies)

# ===================== CONTENT RECOMMENDER =====================
def recommend_content(movie_title, n=5):
    matches = movies[movies["title"] == movie_title]
    if len(matches) == 0:
        return []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:n+1]:
        recommendations.append(movies.iloc[i[0]]["title"])

    return recommendations

# ===================== GENRE RECOMMENDER =====================
def recommend_genre(genre, n=5):
    filtered = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    if len(filtered) == 0:
        return []
    return filtered.sample(min(n, len(filtered)))["title"].tolist()

# ===================== UI =====================
rec_type = st.radio(
    "Choose Recommendation Type:",
    ["Content-Based", "Genre-Based"]
)

st.divider()

# ===================== CONTENT UI =====================
if rec_type == "Content-Based":
    movie_list = sorted(movies["title"].unique())
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    if st.button("Recommend"):
        recs = recommend_content(selected_movie)

        st.subheader("Recommended Movies:")
        if len(recs) == 0:
            st.warning("No recommendations found.")
        else:
            for m in recs:
                st.write("⭐", m)

# ===================== GENRE UI =====================
elif rec_type == "Genre-Based":
    all_genres = set()
    for g in movies["genres"]:
        for x in str(g).split("|"):
            if x.strip():
                all_genres.add(x.strip())

    genre = st.selectbox("Choose a genre:", sorted(all_genres))

    if st.button("Recommend"):
        recs = recommend_genre(genre)

        st.subheader("Recommended Movies:")
        if len(recs) == 0:
            st.warning("No movies found for this genre.")
        else:
            for m in recs:
                st.write("⭐", m)

# ===================== FOOTER =====================
st.divider()
st.caption("Built by Sathvikha Reddy • AI Movie Recommendation System")

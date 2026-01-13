import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD DATA =================

@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

movies = load_data()

# Fill missing values
movies["overview"] = movies["overview"].fillna("")
movies["title"] = movies["title"].fillna("")

# ================= CONTENT BASED MODEL =================

cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movies["overview"])
similarity = cosine_similarity(count_matrix)

# Reset index to avoid mismatch bugs
movies = movies.reset_index(drop=True)

# ================= FUNCTIONS =================

def recommend_content(movie_name):
    if movie_name not in movies["title"].values:
        return []

    index = movies[movies["title"] == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in distances[1:11]:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# ================= STREAMLIT UI =================

st.set_page_config(page_title="AI Movie Recommendation System", layout="centered")

st.title("🎬 AI Movie Recommendation System")

st.write("Select recommendation type:")

rec_type = st.radio(
    label="Recommendation Type",   # ✅ FIXED: not empty anymore
    options=["Content Based", "Popularity Based"]
)

st.write("---")

if rec_type == "Content Based":
    st.subheader("🎯 Content Based Recommendation")

    movie_list = sorted(movies["title"].unique())
    selected_movie = st.selectbox("Select a movie", movie_list)

    if st.button("Recommend"):
        with st.spinner("Finding similar movies..."):
            recs = recommend_content(selected_movie)

        if len(recs) == 0:
            st.error("Movie not found in database.")
        else:
            st.success("Recommended Movies:")
            for i, movie in enumerate(recs, 1):
                st.write(f"{i}. {movie}")

elif rec_type == "Popularity Based":
    st.subheader("🔥 Popular Movies")

    if "vote_count" in movies.columns:
        popular = movies.sort_values(by="vote_count", ascending=False).head(10)
    else:
        popular = movies.head(10)

    for i, movie in enumerate(popular["title"], 1):
        st.write(f"{i}. {movie}")

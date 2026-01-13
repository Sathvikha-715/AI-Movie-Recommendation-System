import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# ===================== LOAD DATA =====================
movies = pd.read_csv("movies.csv")  # Columns: movieId,title,genres
ratings = pd.read_csv("ratings.csv")  # Columns: userId,movieId,rating,timestamp

# ===================== CONTENT-BASED SETUP =====================
count = CountVectorizer(token_pattern='[a-zA-Z0-9]+')
genre_matrix = count.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# ===================== HELPER FUNCTIONS =====================

# Content-Based Recommendation (smart search + dropdown)
def content_recommend(movie_title, n=5, exact=False):
    if exact:
        if movie_title not in movies['title'].values:
            return ["Movie not found in dataset!"]
        idx = movies[movies['title'] == movie_title].index[0]
    else:
        movie_title = movie_title.lower()
        matches = movies[movies['title'].str.lower().str.contains(movie_title)]
        if matches.empty:
            return ["Movie not found in dataset!"]
        actual_title = matches.iloc[0]['title']
        idx = movies[movies['title'] == actual_title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:n+1]]
    return movies['title'].iloc[movie_indices].tolist()

# Collaborative Filtering (SVD)
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
model = SVD()
model.fit(trainset)

def collab_recommend(user_id, n=5):
    movie_ids = ratings['movieId'].unique()
    predictions = [model.predict(user_id, mid) for mid in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = []
    for p in predictions[:n]:
        title = movies[movies['movieId'] == p.iid]['title'].values
        if len(title) > 0:
            top_movies.append(title[0])
    return top_movies

# Hybrid Recommendation
def hybrid_recommend(movie_title, user_id, n=5):
    content_movies = content_recommend(movie_title, n, exact=True)
    collab_movies = collab_recommend(user_id, n)
    hybrid = list(dict.fromkeys(content_movies + collab_movies))  # remove duplicates
    return hybrid[:n]

# Genre-Based Recommendation
all_genres = set()
for g in movies['genres']:
    for genre in g.split('|'):
        all_genres.add(genre)
all_genres = sorted(list(all_genres))

def recommend_by_genre(selected_genre, n=10, min_rating=0):
    genre_movies = movies[movies['genres'].str.contains(selected_genre)]
    popular = ratings.merge(genre_movies, on="movieId")
    top = popular.groupby("title")['rating'].mean()
    top = top[top >= min_rating].sort_values(ascending=False).head(n)
    return top.index.tolist()

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommendation System")

option = st.radio("Choose Recommendation Type:", 
                  ["Content-Based", "Collaborative Filtering", "Hybrid", "Genre-Based"])

# -------------------- CONTENT-BASED --------------------
if option == "Content-Based":
    movie_name = st.selectbox("Choose a movie:", movies['title'].tolist())
    if st.button("Recommend"):
        results = content_recommend(movie_name, exact=True)
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("⭐", movie)

# -------------------- COLLABORATIVE FILTERING --------------------
elif option == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID (1-610):", min_value=1, max_value=610, step=1)
    if st.button("Recommend"):
        results = collab_recommend(user_id)
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("⭐", movie)

# -------------------- HYBRID --------------------
elif option == "Hybrid":
    movie_name = st.selectbox("Choose a movie:", movies['title'].tolist())
    user_id = st.number_input("Enter User ID (1-610):", min_value=1, max_value=610, step=1)
    if st.button("Recommend"):
        results = hybrid_recommend(movie_name, user_id)
        st.subheader("Recommended Movies (Hybrid):")
        for movie in results:
            st.write("⭐", movie)

# -------------------- GENRE-BASED --------------------
elif option == "Genre-Based":
    genre = st.selectbox("Select a Genre:", all_genres)
    min_rating = st.slider("Minimum Rating:", 0.5, 5.0, 3.0, 0.5)
    if st.button("Recommend"):
        results = recommend_by_genre(genre, n=10, min_rating=min_rating)
        st.subheader(f"Top {genre} Movies:")
        for movie in results:
            st.write("⭐", movie)

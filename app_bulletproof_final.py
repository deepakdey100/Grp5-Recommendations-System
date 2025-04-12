
def show_book_cover(isbn):
    url = f"http://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
    st.image(url, width=100)


import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

st.markdown("""
# 📚✨ Book Recommender
Welcome to your personalized book buddy — explore recommendations by users, ratings, and content!
""")
st.caption("User, Item, and Content-Based Recommendations")

# Load and merge datasets
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv", encoding='latin-1')
    ratings = pd.read_csv("Ratings.csv", encoding='latin-1')
    users = pd.read_csv("Users.csv", encoding='latin-1', sep=',', engine='python', on_bad_lines='skip')
    merged = pd.merge(ratings, books, on="ISBN")
    return merged

df = load_data()

# Filter active users and popular books
user_counts = df['User-ID'].value_counts()
book_counts = df['ISBN'].value_counts()

df = df[df['User-ID'].isin(user_counts[user_counts >= 50].index)]
df = df[df['ISBN'].isin(book_counts[book_counts >= 100].index)]

# User-item matrix
user_item_matrix = df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
user_sparse = csr_matrix(user_item_matrix.values)
item_sparse = csr_matrix(user_item_matrix.T.values)

# Nearest neighbor models
user_model = NearestNeighbors(metric='cosine', algorithm='brute')
user_model.fit(user_sparse)

item_model = NearestNeighbors(metric='cosine', algorithm='brute')
item_model.fit(item_sparse)

# UI Mode selection
mode = st.radio("Choose Recommendation Type:", ["User-Based", "Item-Based", "Content-Based"])

if mode == "User-Based":
    user_ids = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Select User-ID:", user_ids)
    user_index = user_ids.index(selected_user)

    distances, indices = user_model.kneighbors(user_sparse[user_index], n_neighbors=6)
    similar_users = [user_ids[i] for i in indices.flatten()[1:]]
    st.subheader("👥 Similar Users")
    st.write(similar_users)

elif mode == "Item-Based":
    book_ids = user_item_matrix.columns.tolist()
    selected_book = st.selectbox("Select ISBN:", book_ids)
    book_index = book_ids.index(selected_book)

    distances, indices = item_model.kneighbors(item_sparse[book_index], n_neighbors=6)
    similar_books = [book_ids[i] for i in indices.flatten()[1:]]

    st.subheader("📖 Similar Books")
    for isbn in similar_books:
        title_row = df[df['ISBN'] == isbn]['Book-Title']
        title = title_row.iloc[0] if not title_row.empty else "Unknown Title"
        col1, col2 = st.columns([1, 4])
        with col1:
            show_book_cover(isbn)
        with col2:
            st.markdown(f"**{title}**\n`ISBN:` {isbn}")

elif mode == "Content-Based":
    st.info("Building TF-IDF Model on Top 1,000 Books...")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    top_books = df['ISBN'].value_counts().head(1000).index
    cb_df = df[df['ISBN'].isin(top_books)].drop_duplicates(subset='Book-Title').reset_index(drop=True)
    cb_df['Book-Title'] = cb_df['Book-Title'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(cb_df['Book-Title'].astype(str))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    st.success("TF-IDF Model Ready.")

    selected_title = st.selectbox("Select a Book Title:", cb_df['Book-Title'].tolist())

    indices = pd.Series(cb_df.index, index=cb_df['Book-Title']).drop_duplicates()
    idx = indices[selected_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]

    st.subheader("📘 Content-Based Recommendations")
    for i in book_indices:
        row = cb_df.iloc[i]
        col1, col2 = st.columns([1, 4])
        with col1:
            show_book_cover(row["ISBN"])
        with col2:
            st.markdown(f"**{row['Book-Title']}**\n`ISBN:` {row['ISBN']}")

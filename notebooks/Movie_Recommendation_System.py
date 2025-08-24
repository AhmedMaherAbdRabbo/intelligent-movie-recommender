"""## Import Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import warnings
warnings.filterwarnings('ignore')

"""## Load and Explore Data"""

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies Dataset Shape:", movies.shape)
print("Ratings Dataset Shape:", ratings.shape)

movies.head()

ratings.head()

"""## Data Preprocessing"""

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Process movies data
movies_clean = movies.copy()
movies_clean['title'] = movies_clean['title'].apply(clean_title)
movies_clean['genres'] = movies_clean['genres'].str.split('|')

movies_clean = movies_clean[~movies_clean['genres'].apply(lambda x: '(no genres listed)' in x)]

movies_clean['genres_text'] = movies_clean['genres'].apply(lambda x: ' '.join(x))

ratings_clean = ratings.drop(['timestamp'], axis=1)

print(f"Clean Movies Dataset: {movies_clean.shape[0]} movies")
print(f"Clean Ratings Dataset: {ratings_clean.shape[0]} ratings")
print(f"Unique Users: {ratings_clean['userId'].nunique()}")

"""## Exploratory Data Analysis"""

plt.figure(figsize=(10, 6))
ratings_clean['rating'].hist(bins=10, color='skyblue', alpha=0.7)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
all_genres = [genre for genres_list in movies_clean['genres'] for genre in genres_list]
genre_counts = pd.Series(all_genres).value_counts().head(10)
genre_counts.plot(kind='bar', color='lightcoral')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## Create Combined Dataset"""

combined_data = ratings_clean.merge(movies_clean[['movieId', 'title', 'genres_text']], on='movieId')
print("Combined Dataset Shape:", combined_data.shape)

combined_data.head()

"""## Content-Based Recommendation System"""

class ContentBasedRecommender:
    def __init__(self, movies_data):
        self.movies_data = movies_data
        self.setup_vectorizers()

    def setup_vectorizers(self):
        # Title-based vectorizer
        self.title_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        self.title_tfidf = self.title_vectorizer.fit_transform(self.movies_data['title'])

        # Genre-based vectorizer
        self.genre_vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.genre_tfidf = self.genre_vectorizer.fit_transform(self.movies_data['genres_text'])

    def search_by_title(self, title, top_n=5):
        title = clean_title(title)
        query_vec = self.title_vectorizer.transform([title])
        similarity = cosine_similarity(query_vec, self.title_tfidf).flatten()
        indices = np.argpartition(similarity, -top_n)[-top_n:]
        results = self.movies_data.iloc[indices][::-1]
        return results[['movieId', 'title', 'genres']]

    def recommend_by_genre(self, genres_text, top_n=10):
        query_vec = self.genre_vectorizer.transform([genres_text])
        similarity = cosine_similarity(query_vec, self.genre_tfidf).flatten()
        indices = np.argpartition(similarity, -top_n)[-top_n:]
        results = self.movies_data.iloc[indices][::-1]
        return results[['movieId', 'title', 'genres']]

content_recommender = ContentBasedRecommender(movies_clean)

"""## Collaborative Filtering Recommendation System"""

class CollaborativeRecommender:
    def __init__(self, combined_data):
        self.combined_data = combined_data
        self.movies_data = combined_data[['movieId', 'title', 'genres_text']].drop_duplicates()

    def get_user_similarity_scores(self, movie_id, min_rating=4.0):
        # Find users who rated this movie highly
        similar_users = self.combined_data[
            (self.combined_data['movieId'] == movie_id) &
            (self.combined_data['rating'] >= min_rating)
        ]['userId'].unique()

        if len(similar_users) == 0:
            return pd.Series(dtype=float)

        # Get movies these users also liked
        similar_user_movies = self.combined_data[
            (self.combined_data['userId'].isin(similar_users)) &
            (self.combined_data['rating'] >= min_rating)
        ]['movieId'].value_counts(normalize=True)

        # Get general popularity of these movies
        all_user_movies = self.combined_data[
            (self.combined_data['movieId'].isin(similar_user_movies.index)) &
            (self.combined_data['rating'] >= min_rating)
        ]['movieId'].value_counts(normalize=True)

        # Calculate recommendation score
        scores = pd.DataFrame({
            'similar_users': similar_user_movies,
            'all_users': all_user_movies
        }).fillna(0)

        scores['recommendation_score'] = np.where(
            scores['all_users'] > 0,
            scores['similar_users'] / scores['all_users'],
            0
        )

        return scores['recommendation_score'].sort_values(ascending=False)

    def recommend_movies(self, movie_title, top_n=10):
        movie_matches = content_recommender.search_by_title(movie_title, 1)
        if movie_matches.empty:
            return pd.DataFrame()

        movie_id = movie_matches.iloc[0]['movieId']
        scores = self.get_user_similarity_scores(movie_id)
        if scores.empty:
            return pd.DataFrame()

        recommendations = scores.head(top_n).reset_index()
        recommendations.columns = ['movieId', 'score']
        recommendations = recommendations.merge(self.movies_data, on='movieId')

        return recommendations[['title', 'genres_text', 'score']].round(4)

collab_recommender = CollaborativeRecommender(combined_data)

"""## Hybrid Recommendation System"""

class HybridRecommender:
    def __init__(self, content_recommender, collab_recommender, combined_data):
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
        self.combined_data = combined_data

    def hybrid_recommend(self, movie_title, top_n=10, content_weight=0.3, collab_weight=0.7):
        movie_matches = self.content_rec.search_by_title(movie_title, 1)
        if movie_matches.empty:
            print(f"No movie found matching '{movie_title}'")
            return pd.DataFrame()

        movie_id = movie_matches.iloc[0]['movieId']
        selected_movie = movie_matches.iloc[0]

        print(f"Selected Movie: {selected_movie['title']}")
        print(f"Genres: {', '.join(selected_movie['genres'])}")
        print("-" * 50)

        genres_text = ' '.join(selected_movie['genres'])
        content_recs = self.content_rec.recommend_by_genre(genres_text, top_n*2)
        content_scores = pd.DataFrame({
            'movieId': content_recs['movieId'],
            'content_score': np.linspace(1.0, 0.1, len(content_recs))
        })

        collab_scores = self.collab_rec.get_user_similarity_scores(movie_id)
        if not collab_scores.empty:
            collab_df = pd.DataFrame({
                'movieId': collab_scores.index,
                'collab_score': collab_scores.values
            })
        else:
            collab_df = pd.DataFrame(columns=['movieId', 'collab_score'])

        all_movies = pd.merge(content_scores, collab_df, on='movieId', how='outer').fillna(0)
        all_movies['hybrid_score'] = (
            content_weight * all_movies['content_score'] +
            collab_weight * all_movies['collab_score']
        )

        top_recommendations = all_movies.nlargest(top_n, 'hybrid_score')

        movies_info = self.combined_data[['movieId', 'title', 'genres_text']].drop_duplicates()
        results = top_recommendations.merge(movies_info, on='movieId')

        results = results[results['movieId'] != movie_id]

        return results[['title', 'genres_text', 'hybrid_score', 'content_score', 'collab_score']].head(top_n)

hybrid_recommender = HybridRecommender(content_recommender, collab_recommender, combined_data)

"""## Test Recommendation Systems"""

def test_recommendations(movie_title):
    print(f"MOVIE RECOMMENDATIONS FOR: '{movie_title}'")
    print("-" * 60)

    # Content-based
    print("\nCONTENT-BASED RECOMMENDATIONS:")
    movie_matches = content_recommender.search_by_title(movie_title, 1)
    if not movie_matches.empty:
        selected_movie = movie_matches.iloc[0]
        genres_text = ' '.join(selected_movie['genres'])
        content_recs = content_recommender.recommend_by_genre(genres_text, 5)
        for i, (_, movie) in enumerate(content_recs.iterrows(), 1):
            print(f"{i}. {movie['title']} - {', '.join(movie['genres'])}")
    else:
        print("No content-based recommendations available")

    # Collaborative
    print("\nCOLLABORATIVE FILTERING RECOMMENDATIONS:")
    collab_recs = collab_recommender.recommend_movies(movie_title, 5)
    if not collab_recs.empty:
        for i, (_, movie) in enumerate(collab_recs.iterrows(), 1):
            print(f"{i}. {movie['title']} (Score: {movie['score']:.3f}) - {movie['genres_text']}")
    else:
        print("No collaborative recommendations available")

    # Hybrid
    print("\nHYBRID RECOMMENDATIONS:")
    hybrid_recs = hybrid_recommender.hybrid_recommend(movie_title, 5)
    if not hybrid_recs.empty:
        for i, (_, movie) in enumerate(hybrid_recs.iterrows(), 1):
            print(f"{i}. {movie['title']} (Score: {movie['hybrid_score']:.3f}) - {movie['genres_text']}")
    else:
        print("No hybrid recommendations available")


# Test
test_movies = ["Toy Story", "The Matrix", "Titanic", "Star Wars"]
for movie in test_movies:
    test_recommendations(movie)
    print("\n" + "="*80 + "\n")

"""## Evaluation Metrics"""

def evaluate_recommendations():
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 50)

    # Dataset statistics
    print("DATASET STATISTICS:")
    print(f"Total Movies: {movies_clean.shape[0]:,}")
    print(f"Total Ratings: {ratings_clean.shape[0]:,}")
    print(f"Total Users: {ratings_clean['userId'].nunique():,}")
    print(f"Average Rating: {ratings_clean['rating'].mean():.2f}")
    print(f"Rating Range: {ratings_clean['rating'].min():.1f} - {ratings_clean['rating'].max():.1f}")

    all_genres = set([genre for genres_list in movies_clean['genres'] for genre in genres_list])
    print(f"Total Genres: {len(all_genres)}")

    # Movie popularity
    movie_rating_counts = combined_data.groupby('movieId')['rating'].count()
    print("\nMOVIE POPULARITY:")
    print(f"Movies with 1 rating: {(movie_rating_counts == 1).sum():,}")
    print(f"Movies with 10+ ratings: {(movie_rating_counts >= 10).sum():,}")
    print(f"Movies with 100+ ratings: {(movie_rating_counts >= 100).sum():,}")

    # User activity
    user_rating_counts = combined_data.groupby('userId')['rating'].count()
    print("\nUSER ACTIVITY:")
    print(f"Users with 1 rating: {(user_rating_counts == 1).sum():,}")
    print(f"Users with 10+ ratings: {(user_rating_counts >= 10).sum():,}")
    print(f"Users with 100+ ratings: {(user_rating_counts >= 100).sum():,}")

# Run evaluation
evaluate_recommendations()

"""## Save models"""

import pickle

# Save Content-Based model
with open("content_recommender.pkl", "wb") as f:
    pickle.dump(content_recommender, f)

# Save Collaborative model
with open("collab_recommender.pkl", "wb") as f:
    pickle.dump(collab_recommender, f)

# Save Hybrid model
with open("hybrid_recommender.pkl", "wb") as f:
    pickle.dump(hybrid_recommender, f)

print("Models have been saved successfully!")


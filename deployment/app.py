from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import logging
import re
from werkzeug.exceptions import BadRequest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Utils ==============
def clean_title(title):
    """Clean movie titles by removing special characters"""
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# ============== Model Classes ==============
class ContentBasedRecommender:
    def __init__(self, movies_data):
        self.movies_data = movies_data
        self.setup_vectorizers()
    
    def setup_vectorizers(self):
        self.title_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        self.title_tfidf = self.title_vectorizer.fit_transform(self.movies_data['title'])
        
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

class CollaborativeRecommender:
    def __init__(self, combined_data):
        self.combined_data = combined_data
        self.movies_data = combined_data[['movieId', 'title', 'genres_text']].drop_duplicates()
    
    def get_user_similarity_scores(self, movie_id, min_rating=4.0):
        similar_users = self.combined_data[
            (self.combined_data['movieId'] == movie_id) & 
            (self.combined_data['rating'] >= min_rating)
        ]['userId'].unique()
        
        if len(similar_users) == 0:
            return pd.Series(dtype=float)
        
        similar_user_movies = self.combined_data[
            (self.combined_data['userId'].isin(similar_users)) & 
            (self.combined_data['rating'] >= min_rating)
        ]['movieId'].value_counts(normalize=True)
        
        all_user_movies = self.combined_data[
            (self.combined_data['movieId'].isin(similar_user_movies.index)) & 
            (self.combined_data['rating'] >= min_rating)
        ]['movieId'].value_counts(normalize=True)
        
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
        if hasattr(self, 'content_recommender'):
            movie_matches = self.content_recommender.search_by_title(movie_title, 1)
        else:
            movie_matches = self.movies_data[
                self.movies_data['title'].str.contains(clean_title(movie_title), case=False, na=False)
            ].head(1)
            
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

# ============== Globals ==============
content_recommender = None
collab_recommender = None
hybrid_recommender = None
movies_clean = None
combined_data = None

# Popular movies list
POPULAR_MOVIES = [
    "Toy Story", "The Matrix", "Titanic", "Star Wars", "Shrek",
    "The Dark Knight", "Forrest Gump", "Pulp Fiction", "The Lion King"
]

# ============== Load Models ==============
def load_models():
    global content_recommender, collab_recommender, hybrid_recommender, movies_clean, combined_data
    try:
        base_path = os.path.join(os.path.dirname(__file__), "models")
        
        with open(os.path.join(base_path, 'content_recommender.pkl'), 'rb') as f:
            content_recommender = pickle.load(f)
            
        with open(os.path.join(base_path, 'collab_recommender.pkl'), 'rb') as f:
            collab_recommender = pickle.load(f)
            
        with open(os.path.join(base_path, 'hybrid_recommender.pkl'), 'rb') as f:
            hybrid_recommender = pickle.load(f)
            
        logger.info("✅ All models loaded successfully!")
        
        movies_clean = content_recommender.movies_data
        combined_data = collab_recommender.combined_data
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False

# ============== Routes ==============
@app.route('/')
def index():
    return render_template('index.html', popular_movies=POPULAR_MOVIES)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': all(model is not None for model in [content_recommender, collab_recommender, hybrid_recommender])
    })

@app.route('/stats')
def stats():
    try:
        if movies_clean is None or combined_data is None:
            return jsonify({"error": "Models not loaded"}), 500
            
        total_movies = len(movies_clean)
        total_ratings = len(combined_data)
        total_users = combined_data['userId'].nunique() if 'userId' in combined_data.columns else 0
        avg_rating = combined_data['rating'].mean() if 'rating' in combined_data.columns else 0
        
        # إصلاح مشكلة genres - التعامل مع الأنواع المختلفة
        all_genres = set()
        for genres in movies_clean['genres']:
            if isinstance(genres, str):
                # إذا كان نص، قسمه بـ |
                for genre in genres.split('|'):
                    if genre.strip():
                        all_genres.add(genre.strip())
            elif isinstance(genres, list):
                # إذا كان قائمة، أضف كل عنصر
                for genre in genres:
                    if genre and str(genre).strip():
                        all_genres.add(str(genre).strip())
        
        total_genres = len(all_genres)
        
        return jsonify({
            "total_movies": int(total_movies),
            "total_ratings": int(total_ratings), 
            "total_users": int(total_users),
            "avg_rating": round(float(avg_rating), 2) if avg_rating > 0 else "N/A",
            "total_genres": int(total_genres)
        })
    except Exception as e:
        logger.error(f"Error in /stats: {str(e)}")
        return jsonify({"error": "Failed to load stats"}), 500

@app.route('/search_movie', methods=['POST'])
def search_movie():
    """Search for movies - used for finding movies to select"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        movie_title = data.get("title", "").strip()

        if not movie_title:
            return jsonify({"error": "⚠️ No movie title provided"}), 400

        # Search for movies by title
        results = content_recommender.search_by_title(movie_title, 5)
        
        if results is None or results.empty:
            return jsonify({"results": [], "message": f"No movies found for '{movie_title}'"}), 200

        return jsonify({"results": results.to_dict(orient="records")}), 200

    except Exception as e:
        logger.error(f"❌ Error in search_movie: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations for a selected movie"""
    try:
        data = request.get_json(force=True, silent=True) or {}

        movie_title = data.get("movie_title", "").strip()
        rec_type = data.get("type", "hybrid")
        top_n = int(data.get("num_recommendations", 8))

        if not movie_title:
            return jsonify({"error": "⚠️ No movie title provided"}), 400

        # Get movie details first
        movie_matches = content_recommender.search_by_title(movie_title, 1)
        if movie_matches.empty:
            return jsonify({"error": f"Movie '{movie_title}' not found"}), 404
            
        selected_movie = movie_matches.iloc[0]

        # Generate recommendations based on type
        if rec_type == "content":
            genres_text = ' '.join(selected_movie['genres'])
            results = content_recommender.recommend_by_genre(genres_text, top_n)
            # Add score column for consistency
            results['score'] = np.linspace(1.0, 0.1, len(results))
            results = results[['title', 'genres', 'score']]
            
        elif rec_type == "collaborative":
            results = collab_recommender.recommend_movies(movie_title, top_n)
            if not results.empty:
                results = results[['title', 'genres_text', 'score']]
                results.rename(columns={'genres_text': 'genres'}, inplace=True)
            
        else:  # hybrid
            results = hybrid_recommender.hybrid_recommend(movie_title, top_n)
            if not results.empty:
                results = results[['title', 'genres_text', 'hybrid_score', 'content_score', 'collab_score']]
                results.rename(columns={
                    'genres_text': 'genres', 
                    'hybrid_score': 'score'
                }, inplace=True)

        # Format results
        if results is None or results.empty:
            recommendations = []
        else:
            recommendations = []
            for _, row in results.iterrows():
                rec = {
                    'title': row['title'],
                    'genres': row['genres'].split('|') if isinstance(row['genres'], str) else row['genres'],
                    'score': float(row['score']) if 'score' in row else None
                }
                
                # Add hybrid-specific scores
                if rec_type == "hybrid" and 'content_score' in row and 'collab_score' in row:
                    rec['content_score'] = float(row['content_score'])
                    rec['collab_score'] = float(row['collab_score'])
                    
                recommendations.append(rec)

        return jsonify({
            "movie_details": {
                "title": selected_movie['title'],
                "genres": selected_movie['genres'].split('|') if isinstance(selected_movie['genres'], str) else selected_movie['genres'],
                "movieId": int(selected_movie['movieId'])
            },
            "recommendations": recommendations,
            "type": rec_type,
            "count": len(recommendations)
        }), 200

    except Exception as e:
        logger.error(f"❌ Error in recommend: {e}")
        return jsonify({"error": str(e)}), 500

# ============== Run ==============
if __name__ == '__main__':
    print("🔄 Loading movie recommendation system...")

    if load_models():
        print("✅ Models loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check models folder.")
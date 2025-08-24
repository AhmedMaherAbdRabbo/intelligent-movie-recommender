# Movie Recommendation System 🎬

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📊 Project Overview

This project implements a **comprehensive movie recommendation system** using multiple machine learning approaches. By analyzing movie metadata, user ratings, and viewing patterns, we develop three distinct recommendation engines: Content-Based, Collaborative Filtering, and Hybrid approaches to provide personalized movie suggestions.

### 🎯 Key Objectives
✔ Implement Content-Based Filtering using TF-IDF vectorization on movie titles and genres

✔ Develop Collaborative Filtering based on user similarity and rating patterns

✔ Create a Hybrid system combining both approaches for enhanced recommendations

✔ Build an interactive web application using Flask for real-time recommendations

✔ Process and clean large-scale movie and rating datasets (62K+ movies, 25M+ ratings)

✔ Implement efficient similarity calculations using cosine similarity metrics

✔ Deploy scalable recommendation models with pickle serialization

## 📁 Dataset Information

### Data Source
- **Dataset**: [Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)
- **Movies Dataset**: `movies.csv` (62,423 movies with 3 features)
- **Ratings Dataset**: `ratings.csv` (25,000,095 ratings with 4 features)

### Key Features
| Feature | Description | Type |
|---------|-------------|------|
| **Movies Dataset** |
| movieId | Unique movie identifier | Numeric |
| title | Movie title with year | Text |
| genres | Movie genres (pipe-separated) | Categorical |
| **Ratings Dataset** |
| userId | Unique user identifier | Numeric |
| movieId | Movie identifier (foreign key) | Numeric |
| rating | User rating (0.5-5.0 scale) | Numeric |
| timestamp | Rating timestamp | Numeric |

## 🗂️ Project Structure

```
Movie-Recommendation-System/
│
├── 📁 data/
│   ├── movies.csv                  # Movie metadata (Download from Kaggle)
│   └── ratings.csv                 # User ratings (Download from Kaggle)
│   └── README.md                   # Data download instructions
│
├── 📁 models/                      # Trained Models (Generated after training)
│   ├── collab_recommender.pkl      # Collaborative Filtering model
│   ├── content_recommender.pkl     # Content-Based model
│   └── hybrid_recommender.pkl      # Hybrid Recommendation model
│
├── 📁 notebooks/
│   ├── Movie_Recommendation_System.ipynb
│   └── Movie_Recommendation_System.py
│
├── 📁 deployment/                  # Web Application
│   ├── app.py                     # Flask application server
│   ├── 📁 static/                 # Frontend assets
│   │   ├── 📁 css/
│   │   │   └── styles.css
│   │   └── 📁 js/
│   │       └── script.js
│   ├── 📁 templates/              # HTML templates
│   │   └── index.html
│   └── 📁 models/                 # Deployed Models
│       ├── collab_recommender.pkl
│       ├── content_recommender.pkl
│       └── hybrid_recommender.pkl
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 🎯 Recommendation Approaches

### 1. Content-Based Filtering 📚
- **Method**: TF-IDF vectorization on movie titles and genres
- **Similarity**: Cosine similarity between movie features
- **Strength**: Works well for new users, genre-specific recommendations
- **Use Case**: "Movies similar to Toy Story"

### 2. Collaborative Filtering 👥
- **Method**: User-item rating matrix analysis
- **Similarity**: User behavior patterns and preferences
- **Strength**: Discovers hidden patterns, handles diverse tastes
- **Use Case**: "Users who liked this movie also enjoyed..."

### 3. Hybrid System 🔄
- **Method**: Weighted combination of Content-Based (30%) + Collaborative (70%)
- **Advantage**: Combines strengths of both approaches
- **Performance**: Best overall recommendation quality
- **Features**: Balanced content relevance and user preference alignment

## 🔍 Key Features & Functionality

### Data Processing Pipeline
1. **Title Cleaning**: Remove special characters and normalize text
2. **Genre Processing**: Split pipe-separated genres into lists
3. **Rating Filtering**: Handle missing values and outliers
4. **Feature Engineering**: Create combined text features for vectorization

### Recommendation Engine Features
- **Smart Search**: Fuzzy matching for movie title searches
- **Scalable Architecture**: Efficient similarity calculations for large datasets
- **Flexible Scoring**: Adjustable weights for hybrid recommendations
- **Real-time Inference**: Fast recommendation generation (<1 second)

### Web Application Features
- **Interactive UI**: Modern, responsive web interface
- **Multiple Recommendation Types**: Switch between Content, Collaborative, and Hybrid
- **Movie Search**: Auto-complete movie search functionality
- **Detailed Results**: Show recommendation scores and movie genres
- **System Statistics**: Dataset insights and performance metrics

## 📈 Performance & Statistics

### Dataset Statistics
- **Total Movies**: 62,423 unique films
- **Total Ratings**: 25,000,095 user ratings
- **Total Users**: 162,541 unique users
- **Average Rating**: 3.5/5.0
- **Rating Distribution**: Comprehensive coverage across 0.5-5.0 scale
- **Genre Coverage**: 20+ distinct movie genres

### System Performance
- **Content-Based**: Fast inference, good genre coverage
- **Collaborative**: High accuracy for popular movies
- **Hybrid**: Best overall recommendation quality
- **Response Time**: <1 second for 10 recommendations
- **Memory Efficiency**: Optimized vectorization and storage


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📞 Contact

**Ahmed Maher Abd Rabbo**
- 💼 [LinkedIn](https://www.linkedin.com/in/ahmed-maherr/)
- 📊 [Kaggle](https://kaggle.com/ahmedmaherabdrabbo)
- 📧 Email: ahmedbnmaher1@gmail.com
- 💻 [GitHub](https://github.com/AhmedMaherAbdRabbo)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
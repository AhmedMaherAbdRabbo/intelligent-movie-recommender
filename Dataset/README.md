# Dataset Information 📊

## Download Instructions

Download the dataset from Kaggle:
```bash
kaggle datasets download -d parasharmanas/movie-recommendation-system
unzip movie-recommendation-system.zip
```

**Dataset Link**: [Movie Recommendation System Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

## Files

### 1. `movies.csv` (62,423 rows)
| Column | Description |
|--------|-------------|
| movieId | Unique movie identifier |
| title | Movie title with release year |
| genres | Pipe-separated movie genres |

### 2. `ratings.csv` (25,000,095 rows)
| Column | Description |
|--------|-------------|
| userId | Unique user identifier |
| movieId | Movie identifier |
| rating | User rating (0.5-5.0) |
| timestamp | Rating timestamp |

## Dataset Stats
- **Total Movies**: 62,423
- **Total Ratings**: 25M+
- **Total Users**: 162,541
- **Rating Scale**: 0.5 to 5.0
- **Genres**: 20+ categories

## Usage
After downloading, place the CSV files in this `data/` folder to run the recommendation system.
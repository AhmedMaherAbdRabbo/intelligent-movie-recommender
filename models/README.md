# Trained Models 🤖

## Model Files

After running the training notebook/script, the following model files will be generated:

```
models/
├── content_recommender.pkl      # Content-Based Filtering model
├── collab_recommender.pkl       # Collaborative Filtering model
└── hybrid_recommender.pkl       # Hybrid Recommendation model
```

## File Sizes
- Each model file is approximately **50-800 MB**
- Files are not included in the repository due to size limitations
- Models are automatically saved after training completion

## Training
To generate these models, run:
```bash
python notebooks/Movie_Recommendation_System.py
```

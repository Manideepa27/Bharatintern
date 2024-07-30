import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings data
ratings = pd.read_csv('ratings.csv')

# Load movies data
movies = pd.read_csv('movies.csv')

# Merge ratings with movies data using movie_id
merged_data = pd.merge(ratings, movies, on='movie_id')

# Build user-item matrix
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(merged_data, reader)

# Compute similarity between users
similarity = cosine_similarity(data.user_item_matrix)

# Train KNN model
knn = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
knn.fit(data.build_full_trainset())

# Function to generate recommendations
def get_recommendations(user_id, num_recs):
    # Compute predicted ratings
    predicted_ratings = knn.test(data.build_testset()[user_id])
    # Rank movies by predicted ratings
    ranked_movies = sorted(predicted_ratings, key=lambda x: x.est, reverse=True)
    return ranked_movies[:num_recs]

# Test recommendations
user_id = 1
num_recs = 10
recommendations = get_recommendations(user_id, num_recs)
print(recommendations)

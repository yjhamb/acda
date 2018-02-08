'''
Utility module for working on the movielens movie dataset
'''

import pandas as pd

RATINGS_FILE = "../../../dataset/movielens/ratings.csv"
RATINGS_CONTEXT_FILE = "../../../dataset/movielens/rating_context.csv"
MOVIES_FILE = "../../../dataset/movielens/movies.csv"
TAGS_FILE = "../../../dataset/movielens/tags.csv"

def generate_movie_dataset():
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    ratings_context = pd.merge(ratings, movies, on='movieId', how='left')
    ratings_context.to_csv(RATINGS_CONTEXT_FILE, index=False)
    

def main():
    print("Main method")
    generate_movie_dataset()
    print("Main method complete")

if __name__ == '__main__':
    main()

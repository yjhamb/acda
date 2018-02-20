'''
Functions that operate on the movielens dataset
'''
import os
import pandas as pd
import numpy as np
import sklearn.model_selection as ms

from scipy import sparse
from scipy.stats import bernoulli
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

RATINGS_CONTEXT_FILE = "../../../dataset/movielens/rating_context.csv"

class MovieRatingsData(object):

    def __init__(self):
        self.ratings = pd.read_csv(RATINGS_CONTEXT_FILE)
        ratings_sorted = self.ratings.sort_values(['timestamp'], ascending=True)
        # perform the train-test split
        self.train_ratings, self.test_ratings = ms.train_test_split(ratings_sorted, test_size=0.2, random_state=42)

        # split again to generate CV set
        self.train_ratings, self.cv_ratings = ms.train_test_split(self.train_ratings, test_size=0.25, random_state=42)

        print("Total Ratings:", len(self.ratings))
        print("Train Ratings:", len(self.train_ratings))
        print("CV Ratings:", len(self.cv_ratings))
        print("Test Ratings:", len(self.test_ratings))
        
        self._n_users = len(self.get_users())
        self._n_movies = len(self.get_movies())
        self._n_genres = len(self.get_genres())
        
        print("Total Users:", self._n_users)
        print("Total Movies:", self._n_movies)
        print("Total Genres:", self._n_genres)

        # Convert the sparse event indices to a dense vector
        self._mlb_movie = MultiLabelBinarizer(sparse_output=True)
        self._mlb_movie.fit([self.get_movies()])
        # We need this to get the indices of events
        self._movie_class_to_index = dict(zip(self._mlb_movie.classes_, range(len(self._mlb_movie.classes_))))

        self._user_encoder = LabelEncoder().fit(self.get_users())
        self._genre_encoder = LabelEncoder().fit(self.get_genres())

    @property
    def n_users(self):
        """Return the number of users in the dataset"""
        return self._n_users

    @property
    def n_movies(self):
        """Return the number of movies in the dataset"""
        return self._n_movies

    @property
    def n_genres(self):
        """Return the number of genres in the dataset"""
        return self._n_genres

    def get_user_train_movie_index(self, user_id):
        """
        Get the converted index of user movies
        """
        unique_user_train_movies = self.train_ratings.movieId[self.train_ratings.userId == user_id].unique()
        return [self._movie_class_to_index[i]
                            for i in unique_user_train_movies]

    def get_user_test_movie_index(self, user_id):
        """
        Get the converted index of user movies
        """
        unique_user_test_movies = self.test_ratings.movieId[self.test_ratings.userId == user_id].unique()
        return [self._movie_class_to_index[i]
                            for i in unique_user_test_movies]

    def get_user_cv_movie_index(self, user_id):
        """
        Get the converted index of user events
        """
        unique_user_cv_movies = self.cv_ratings.movieId[self.cv_ratings.userId == user_id].unique()
        return [self._movie_class_to_index[i]
                            for i in unique_user_cv_movies]
    
    def get_user_index(self, user_id):
        """
        Get user index
        """
        return self._user_encoder.transform([user_id])

    def get_users(self):
        return self.ratings.userId.unique()

    def get_movies(self):
        return self.ratings.movieId.unique()

    def get_genres(self):
        genre_set = set()
        genres_list = self.ratings.genres
        for genre in genres_list:
            split_genre = genre.split('|')
            for g in split_genre:
                genre_set.add(g)
        
        return list(genre_set)

    def get_train_users(self):
        return self.train_ratings.userId.unique()

    def get_test_users(self):
        return self.test_ratings.userId.unique()

    def get_cv_users(self):
        return self.cv_ratings.userId.unique()

    def get_train_movies(self):
        return self.train_ratings.movieId.unique()

    def get_test_movies(self):
        return self.test_ratings.movieId.unique()

    def get_user_train_genres(self, user_id):
        """
        Get train user genre indexes
        """
        genre_set = set()
        genres_list = self.train_ratings[self.train_ratings.userId == user_id].genres
        for genre in genres_list:
            split_genre = genre.split('|')
            for g in split_genre:
                genre_set.add(g)
        
        return self._genre_encoder.transform(list(genre_set))

    def get_user_unique_test_movies(self, user_id):
        return self.test_ratings.movieId[self.test_ratings.userId == user_id].unique()


    def get_user_train_movies(self, user_id, negative_count, corrupt_ratio):
        """
        Calls the get_user_movies method with the training data
        """
        return self.get_user_movies(user_id, self.train_ratings, negative_count, corrupt_ratio)


    def get_user_test_movies(self, user_id):
        """
        Calls the get_user_movies method with the test data
        """
        return self.get_user_movies(user_id, self.test_ratings)

    def get_user_movies(self, user_id, df, negative_count=0, corrupt_ratio=0):
        """
        This will get a single users movies (training or test based on input parameter).
        We encode each user with a k-hot encoding, where a 1 if they have rated the item.
        We then sample negative items they have not observed, if neg_ratio > 0.
        Negative items have a target of 0 and positives 1.
        We finally corrupt all the encoded user vectors, if the corrupt_ratio > 0.

        :param user_id: user id in dataframe
        :param df: the dataframe for training or test
        :param negative_count: int, ratio of negative samples
        :param corrupt_ratio: float, [0, 1] the probability of corrupting samples
        :returns: Encoded User Vector, Y Target, item ids
        """
        # Get positive samples
        positives = df.movieId[df.userId == user_id].unique().tolist()

        # Testing...
        if negative_count == 0:
            return self._mlb_movie.transform([positives]).tocoo(), None, None

        x, y, items = [], [], []

        # For each positive we have N negatives
        # i.e. len(positives) * (neg_count+1)
        for i in range(len(positives)):
            # Add a negative example
            for j in range(negative_count):
                neg = self._sample_negative_new(positives, self._mlb_movie.classes_)
                y.append(0) # Negative Target
                items.append(self._movie_class_to_index[neg]) # Negative index
                x.append(positives + [neg]) # add negative pair
            y.append(1)
            x.append(positives)
            items.append(self._movie_class_to_index[positives[i]])

        x = self._mlb_movie.transform(x).tocoo()

        # Sparse Matrix; directly take the data and corrupt it
        if corrupt_ratio > 0:
            x.data = self.corrupt_input(x.data, corrupt_ratio).astype(np.float32)

        return x.astype(np.float32), np.array(y, dtype=np.float32), items


    def sample_negative(self, pos_item_map, max_items):
        """Sample uniformly items that are not observed

        :param pos_item_map: set/list, listing all of the users observed items
        :param max_items: int, item count
        :returns: int negative item id
        """
        while True:
            sample = np.random.randint(max_items)
            if sample in pos_item_map:
                continue
            return sample

    def _sample_negative_new(self, pos_items, all_items):
        """Sample uniformly items that are not observed

        :param pos_item_map: set/list, listing all of the users observed items
        :param max_items: int, item count
        :returns: int negative item id
        """
        while True:
            sample = np.random.choice(all_items)
            if sample in pos_items:
                continue
            return sample


    def sample_negative_on_context(self, df, user_id, count):
        """Sample uniformly items that are not observed

        :param df: Dataframe to be sampled
        :param user_id: user id for which the samples are to be provided
        :param count: negative item count
        :returns: int negative item id
        """

        return df[df.memberId != user_id].sample(count)


    def corrupt_input(self, x, q):
        """
        Corrupt x with probability p by setting it to 0
        else scale it by 1 / (1-p)
        """
        assert x.ndim == 1
        scale = 1.0 / (1.0-q)
        # Probability to remove it
        # p = 1; 1-p = 0
        p = 1-q
        mask = bernoulli.rvs(p, size=x.shape[0])
        # Mask outputs
        x = x * mask
        # Re-scale values
        return x * scale


    def _get_batch(self, df, user_ids, mlb):
        """
        This method creates a single vector for each user where all of their
        events are set to a 1, otherwise its a 0.

        In this case, this user has observed events: 1 and 4
        [1, 0, 0, 1, 0]

        TODO: Should probably abstract this out somewhere

        :param df: pd.DataFrame of test/train
        :param user_ids: list of user ids, this will query the dataframe
        :param mlb: sklearn.MultiLabelBinarizer fitted with the event data
        :returns: np.array of values
        """
        item_ids = [df.movieId[df.movieId == uid].unique() for uid in user_ids
                    if len(df.movieId[df.userId == uid].unique()) > 0]
        return mlb.transform(item_ids), item_ids


def main():
    print("main method")
    ratings = MovieRatingsData()
    print(ratings.get_genres())

if __name__ == '__main__':
    main()

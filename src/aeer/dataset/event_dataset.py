'''
Functions that operate on the event dataset
'''
import os
import pandas as pd
import numpy as np
import sklearn.model_selection as ms

from scipy import sparse
from scipy.stats import bernoulli
from sklearn.preprocessing import MultiLabelBinarizer
import aeer.dataset.user_group_dataset as ug_dataset

rsvp_ny_file = "../../../dataset/rsvp_ny.csv"
rsvp_sfo_file = "../../../dataset/rsvp_sfo.csv"
rsvp_dc_file = "../../../dataset/rsvp_dc.csv"
rsvp_chicago_file = "../../../dataset/rsvp_chicago.csv"


class EventData(object):

    def __init__(self, rsvp_file, user_group_file):
        self.events = pd.read_csv(rsvp_file)
        # sort the event data by event time
        events_sorted = self.events.sort_values(['eventTime'], ascending=True)
        x = events_sorted.drop(['rsvpRating'], axis=1)
        y = events_sorted[['rsvpRating']]

        # perform the train-test split
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(x, y, test_size=0.2, random_state=42)
        
        self._n_users = len(self.get_users())
        self._n_events = len(self.get_events())
        self._n_groups = len(self.get_groups())
        self._n_venues = len(self.get_venues())
        
        self._user_group_data = ug_dataset.UserGroupData(user_group_file)

        # Convert the sparse event indices to a dense vector
        mlb = MultiLabelBinarizer()
        mlb.fit([self.get_events()])
        # We need this to get the indices of events
        self._event_class_to_index = dict(zip(mlb.classes_, range(len(mlb.classes_))))

        # Convert the sparse group indices to a dense vector
        mlb_group = MultiLabelBinarizer()
        mlb_group.fit([self.get_groups()])
        # We need this to get the indices of groups
        self._group_class_to_index = dict(zip(mlb_group.classes_, range(len(mlb_group.classes_))))
        
        # Convert the sparse group indices to a dense vector
        mlb_venue = MultiLabelBinarizer()
        mlb_venue.fit([self.get_venues()])
        # We need this to get the indices of venues
        self._venue_class_to_index = dict(zip(mlb_venue.classes_, range(len(mlb_venue.classes_))))

    @property
    def n_users(self):
        """Return the number of users in the dataset"""
        return self._n_users

    @property
    def n_events(self):
        """Return the number of events in the dataset"""
        return self._n_events

    @property
    def n_groups(self):
        """Return the number of groups in the dataset"""
        return self._n_groups

    @property
    def n_venues(self):
        """Return the number of venues in the dataset"""
        return self._n_venues

    def get_user_test_event_index(self, user_id):
        """
        Get the converted index of user events
        """
        unique_user_test_events = self.test_x.eventId[self.test_x.memberId == user_id].unique()
        return [self._event_class_to_index[i]
                            for i in unique_user_test_events]
        

    def get_user_train_groups(self, user_id):
        """
        Get train user group indexes
        """
        groups = [self._group_class_to_index[i] for i in
                  self.train_x[self.train_x.memberId == user_id].groupId.unique()]
        # get groups based on the membership
        #group_list = []
        #for g in self.train_x[self.train_x.memberId == user_id].groupId.unique():
        #    if self._user_group_data.is_user_in_group(user_id, g):
        #        group_list.append(g)
        #groups = [self._group_class_to_index[i] for i in group_list]
        
        return groups

    def get_user_train_venues(self, user_id):
        """
        Get train user venue indexes
        """
        venues = [self._venue_class_to_index[i] for i in
                  self.train_x[self.train_x.memberId == user_id].venueId.unique()]
        
        return venues

    def get_users(self):
        return self.events.memberId.unique()


    def get_events(self):
        return self.events.eventId.unique()


    def get_venues(self):
        return self.events.venueId.unique()


    def get_groups(self):
        return self.events.groupId.unique()


    def get_train_users(self):
        return self.train_x.memberId.unique()


    def get_test_users(self):
        return self.test_x.memberId.unique()


    def get_train_events(self):
        return self.train_x.eventId.unique()


    def get_test_events(self):
        return self.test_x.eventId.unique()


    def get_user_unique_test_events(self, user_id):
        return self.test_x.eventId[self.test_x.memberId == user_id].unique()


    def get_user_train_events(self, user_id, negative_count, corrupt_ratio):
        """
        Calls the get_user_events method with the training data
        """
        return self.get_user_events(user_id, self.train_x, negative_count, corrupt_ratio)


    def get_user_test_events(self, user_id):
        """
        Calls the get_user_events method with the test data
        """
        return self.get_user_events(user_id, self.test_x)


    def get_user_events(self, user_id, df, negative_count=0, corrupt_ratio=0):
        """
        This will get a single users events (training or test based on input parameter).
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

        event_count = self.n_events

        # Get all positive items
        positives = [self._event_class_to_index[i] for i in df.eventId[df.memberId == user_id].unique()]

        # Sample negative items
        if negative_count > 0:
            negatives = [self.sample_negative(positives, event_count) for _ in range(negative_count)]
        else:
            negatives = []

        input_count = len(positives) + len(negatives)

        # X vector for a single user
        # Duplicate input count times
        x_data = [1.0] * input_count

        # Indices for the items
        cols = positives + negatives
        rows = []
        if negative_count > 0:
            for i in range(input_count):
                rows.extend([i] * input_count)
            x = sparse.coo_matrix((x_data * input_count,
                               (rows, cols * input_count)),
                              shape=(input_count, event_count),
                              dtype=np.float32)
        else:
            rows.extend([0] * input_count)
            x = sparse.coo_matrix((x_data,
                               (rows, cols)),
                              shape=(1, event_count),
                              dtype=np.float32)

        # Negative targets are 0, positives are 1
        y_targets = np.zeros(input_count, dtype=np.float32)
        y_targets[:len(positives)] = 1.0

        # Sparse Matrix; directly take the data and corrupt it
        if corrupt_ratio > 0:
            x.data = self.corrupt_input(x.data, corrupt_ratio).astype(np.float32)

        return x, y_targets, cols


    def get_user_train_events_with_context(self, user_id, negative_count=0, corrupt_ratio=0):
        """
        Calls the get_user_events method with the training data
        """
        return self.get_user_events_with_context(user_id, self.train_x, negative_count, corrupt_ratio)


    def get_user_test_events_with_context(self, user_id):
        """
        Calls the get_user_events method with the test data
        """
        return self.get_user_events_with_context(user_id, self.test_x)


    def get_user_events_with_context(self, user_id, df, negative_count=0, corrupt_ratio=0):
        """
        This will get a single users events (training or test based on input parameter).
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

        event_count = self.n_events
        group_count = self.n_groups
        venue_count = self.n_venues

        # Get all positive items
        positive_samples = df[df.memberId == user_id]
        positive_events = [self._event_class_to_index[i] for i in positive_samples.eventId.unique()]
        positive_groups = [self._group_class_to_index[i] for i in positive_samples.groupId.unique()]
        positive_venues = [self._venue_class_to_index[i] for i in positive_samples.venueId.unique()]

        # Sample negative items
        negative_events = []
        negative_groups = []
        negative_venues = []
        if negative_count > 0:
            negative_samples = self.sample_negative_on_context(df, user_id, negative_count)
            negative_events = [self._event_class_to_index[i] for i in negative_samples.eventId.unique()]
            negative_groups = [self._group_class_to_index[i] for i in negative_samples.groupId.unique()]
            negative_venues = [self._venue_class_to_index[i] for i in negative_samples.venueId.unique()]
            
        input_count = len(positive_events) + len(negative_events)

        # X vector for a single user
        # Duplicate input count times
        x_data = [1.0] * input_count

        # Indices for the items
        cols = positive_events + negative_events
        rows = []
        if negative_count > 0:
            for i in range(input_count):
                rows.extend([i] * input_count)
            x = sparse.coo_matrix((x_data * input_count,
                               (rows, cols * input_count)),
                              shape=(input_count, event_count),
                              dtype=np.float32)
        else:
            rows.extend([0] * input_count)
            x = sparse.coo_matrix((x_data,
                               (rows, cols)),
                              shape=(1, event_count),
                              dtype=np.float32)

        # Negative targets are 0, positives are 1
        y_targets = np.zeros(input_count, dtype=np.float32)
        y_targets[:len(positive_events)] = 1.0

        # Sparse Matrix; directly take the data and corrupt it
        if corrupt_ratio > 0:
            x.data = self.corrupt_input(x.data, corrupt_ratio).astype(np.float32)

        input_group_count = len(positive_groups) + len(negative_groups)
        if input_group_count > 0:
            # create input data vector for the groups
            x_group_data = [1.0] * len(positive_groups) + [0.0] * len(negative_groups)

            # Indices for the items
            group_cols = positive_groups + negative_groups
            group_rows = []
            if negative_count > 0:
                for i in range(input_group_count):
                    group_rows.extend([i] * input_group_count)
                x_group = sparse.coo_matrix((x_group_data * input_group_count,
                                   (group_rows, group_cols * input_group_count)),
                                  shape=(input_group_count, group_count),
                                  dtype=np.float32)
            else:
                group_rows.extend([0] * input_group_count)
                x_group = sparse.coo_matrix((x_group_data,
                                   (group_rows, group_cols)),
                                  shape=(1, group_count),
                                  dtype=np.float32)
            
            # vstack both the event and group matrices
            diff_n_rows = 0
            x_padded = x
            x_group_padded = x_group
            if x.shape[0] > x_group.shape[0]:
                diff_n_rows = x.shape[0] - x_group.shape[0]
                x_group_padded = sparse.vstack((x_group, sparse.csr_matrix((diff_n_rows, x_group.shape[1])))) 
            elif x.shape[0] < x_group.shape[0]:
                diff_n_rows = x_group.shape[0] - x.shape[0]
                x_padded = sparse.vstack((x, sparse.csr_matrix((diff_n_rows, x.shape[1])))) 
        else:
            x_padded = x
            x_group_padded = sparse.coo_matrix(shape=(input_count, group_count), dtype=np.float32)
            
        input_x = sparse.hstack((x_padded, x_group_padded))
        
        input_venue_count = len(positive_venues) + len(negative_venues)
        if input_venue_count > 0:
            # create input data vector for the venues
            x_venue_data = [1.0] * len(positive_venues) + [0.0] * len(negative_venues)

            # Indices for the items
            venue_cols = positive_venues + negative_venues
            venue_rows = []
            if negative_count > 0:
                for i in range(input_venue_count):
                    venue_rows.extend([i] * input_venue_count)
                x_venue = sparse.coo_matrix((x_venue_data * input_venue_count,
                                   (venue_rows, venue_cols * input_venue_count)),
                                  shape=(input_venue_count, venue_count),
                                  dtype=np.float32)
            else:
                venue_rows.extend([0] * input_venue_count)
                x_venue = sparse.coo_matrix((x_venue_data,
                                   (venue_rows, venue_cols)),
                                  shape=(1, venue_count),
                                  dtype=np.float32)
            
            # vstack both the event and group matrices
            diff_n_rows = 0
            x_padded_input = input_x
            x_venue_padded = x_venue
            if input_x.shape[0] > x_venue.shape[0]:
                diff_n_rows = input_x.shape[0] - x_venue.shape[0]
                x_venue_padded = sparse.vstack((x_venue, sparse.csr_matrix((diff_n_rows, x_venue.shape[1])))) 
            elif input_x.shape[0] < x_venue.shape[0]:
                diff_n_rows = x_venue.shape[0] - input_x.shape[0]
                x_padded_input = sparse.vstack((input_x, sparse.csr_matrix((diff_n_rows, input_x.shape[1])))) 
        else:
            x_padded_input = input_x
            x_venue_padded = sparse.coo_matrix(shape=(input_count + input_group_count, venue_count), dtype=np.float32)
            
        input_x_final = sparse.hstack((x_padded_input, x_venue_padded))
        
        return input_x_final, y_targets, cols


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
        item_ids = [df.eventId[df.memberId == uid].unique() for uid in user_ids
                    if len(df.eventId[df.memberId == uid].unique()) > 0]
        return mlb.transform(item_ids), item_ids


def main():
    print("main method")
    print("Current Directory: ", os.getcwd())
    event_data = EventData(rsvp_chicago_file, ug_dataset.user_group_chicago_file)
    users = event_data.get_users()
    events = event_data.get_events()
    print("Users:", len(users))
    print("Events:", len(events))

if __name__ == '__main__':
    main()

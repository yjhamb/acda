'''
Functions that operate on the event dataset
'''
import os
import pandas as pd
import numpy as np
import sklearn.model_selection as ms

from scipy import sparse
from scipy.stats import bernoulli

ny_file_name = "../../../dataset/rsvp_ny.csv"
sfo_file_name = "../../../dataset/rsvp_sfo.csv"
dc_file_name = "../../../dataset/rsvp_dc.csv"
chicago_file_name = "../../../dataset/rsvp_chicago.csv"

class EventData(object):
    
    def __init__(self, file_name):
        self.events = pd.read_csv(file_name)
        # sort the event data by event time
        events_sorted = self.events.sort_values(['eventTime'], ascending=True)
        x = events_sorted.drop(['rsvpRating'], axis=1)
        y = events_sorted[['rsvpRating']]
    
        # perform the train-test split
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(x, y, test_size=0.2, random_state=42)

    
    def get_users(self):
        return self.events.memberId.unique()

    
    def get_events(self):
        return self.events.eventId.unique()


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

    
    def get_user_train_events(self, user_id, class_to_index, negative_count, corrupt_ratio):
        """
        Calls the get_user_events method with the training data
        """
        return self.get_user_events(user_id, self.train_x, class_to_index, negative_count, corrupt_ratio)

    
    def get_user_test_events(self, user_id, class_to_index):
        """
        Calls the get_user_events method with the test data
        """
        return self.get_user_events(user_id, self.test_x, class_to_index)

    
    def get_user_events(self, user_id, df, class_to_index, negative_count=0, corrupt_ratio=0):
        """
        This will get a single users events (training or test based on input parameter). 
        We encode each user with a k-hot encoding, where a 1 if they have rated the item. 
        We then sample negative items they have not observed, if neg_ratio > 0. 
        Negative items have a target of 0 and positives 1. 
        We finally corrupt all the encoded user vectors, if the corrupt_ratio > 0.
    
        :param user_id: user id in dataframe
        :param df: the dataframe for training or test
        :param class_to_index: dictionary that maps item ids to indices
        :param negative_count: int, ratio of negative samples
        :param corrupt_ratio: float, [0, 1] the probability of corrupting samples
        :returns: Encoded User Vector, Y Target, item ids
        """
    
        event_count = len(class_to_index)
    
        # Get all positive items
        positives = [class_to_index[i] for i in df.eventId[df.memberId == user_id].unique()]
    
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
        for i in range(input_count):
            rows.extend([i] * input_count)
    
        x = sparse.coo_matrix((x_data * input_count,
                               (rows, cols * input_count)),
                              shape=(input_count, event_count),
                              dtype=np.float32)
    
        # Negative targets are 0, positives are 1
        y_targets = np.zeros(input_count, dtype=np.float32)
        y_targets[:len(positives)] = 1.0
    
        # Sparse Matrix; directly take the data and corrupt it
        if corrupt_ratio > 0:
            x.data = self.corrupt_input(x.data, corrupt_ratio).astype(np.float32)
        
        return x, y_targets, cols


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
    event_data = EventData(chicago_file_name)
    users = event_data.get_users()
    events = event_data.get_events()
    print("Users:", len(users))
    print("Events:", len(events))

if __name__ == '__main__':
    main()
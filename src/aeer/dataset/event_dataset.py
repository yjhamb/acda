'''
Functions that operate on the dataset
'''
import os
import pandas as pd
import sklearn.model_selection as ms
from sklearn.preprocessing import MultiLabelBinarizer

ny_file_name = "../../../dataset/rsvp_ny.csv"
sfo_file_name = "../../../dataset/rsvp_sfo.csv"
dc_file_name = "../../../dataset/rsvp_dc.csv"
chicago_file_name = "../../../dataset/rsvp_chicago.csv"

class EventData(object):
    
    def __init__(self, file_name):
        self.events = pd.read_csv(file_name)
        
    def split_dataset(self):
        # sort the event data by event time
        events_sorted = self.events.sort_values(['eventTime'], ascending=True)
        x = events_sorted.drop(['rsvpRating'], axis=1)
        y = events_sorted[['rsvpRating']]
    
        # perform the train-test split
        self.train_x, self.test_x, self.train_y, self.test_y = ms.train_test_split(x, y, test_size=0.2, random_state=42)
        return self.train_x, self.test_x, self.train_y, self.test_y
    
    def get_users(self):
        return self.events.memberId.unique()
    
    def get_events(self):
        return self.events.eventId.unique()
    
    def get_test_events(self):
        return self.test_x.eventId.unique()
    
    def get_user_test_events(self, uid):
        return self.test_x.eventId[self.test_x.memberId == uid].unique()
    
    def get_user_test_events_labeled(self):
        # Convert the sparse event indices to a dense vector
        mlb = MultiLabelBinarizer()
        test_events = self.get_test_events()
        mlb.fit([test_events])
        user_test_events = self.get_user_test_events()
        return mlb.transform(user_test_events)
       
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
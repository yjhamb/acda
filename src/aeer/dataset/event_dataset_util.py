'''
Utility methods for the event dataset
'''

import pandas as pd
import sklearn.model_selection as ms

ny_file_name = "../../../dataset/rsvp_ny.csv"
ny_file_name_new = "../../../dataset/rsvp_ny_new.csv"
sfo_file_name = "../../../dataset/rsvp_sfo.csv"
sfo_file_name_new = "../../../dataset/rsvp_sfo_new.csv"
dc_file_name = "../../../dataset/rsvp_dc.csv"
dc_file_name_new = "../../../dataset/rsvp_dc_new.csv"
chicago_file_name = "../../../dataset/rsvp_chicago.csv"
chicago_file_name_new = "../../../dataset/rsvp_chicago_new.csv"

def update_rsvp_data(file_name):
    events = pd.read_csv(file_name)
    print(events.dtypes)
    print(events['rsvpRating'].unique())
    events.loc[events['rsvpResponse'] == 'watching', 'rsvpRating'] = 1
    events.loc[events['rsvpResponse'] == 'waitlist', 'rsvpRating'] = 1
    print(events['rsvpResponse'].unique())
    print(events['rsvpRating'].unique())
    events.to_csv(dc_file_name_new, index=False)

def perform_train_test_split():
    rsvp_file = "../../../dataset/rsvp_chicago.csv"
    train_file = "../../../dataset/rsvp_chicago_train.csv"
    #cv_file = "../../../dataset/rsvp_ny_cv.csv"
    test_file = "../../../dataset/rsvp_chicago_test.csv"
    
    events = pd.read_csv(rsvp_file)
    events_sorted = events.sort_values(['eventTime'], ascending=True)
    
    # perform the train-test split
    train_events, test_events = ms.train_test_split(events_sorted, test_size=0.2, random_state=42)

    # split again to generate CV set
    #train_events, cv_events = ms.train_test_split(train_events, test_size=0.25, random_state=42)
        
    train_events.to_csv(train_file, index=False)
    #cv_events.to_csv(cv_file, index=False)
    test_events.to_csv(test_file, index=False)

def generate_librec_rating_file():
    train_file = "../../../dataset/rsvp_chicago_train.csv"
    train_rating_file = "../../../dataset/rsvp_chicago_train_rating.csv"
    test_file = "../../../dataset/rsvp_chicago_test.csv"
    test_rating_file = "../../../dataset/rsvp_chicago_test_rating.csv"
    
    train_events = pd.read_csv(train_file)
    train_events_ratings = train_events[['memberId', 'eventId', 'rsvpRating']]
    
    test_events = pd.read_csv(test_file)
    test_events_ratings = test_events[['memberId', 'eventId', 'rsvpRating']]
        
    train_events_ratings.to_csv(train_rating_file, index=False)
    test_events_ratings.to_csv(test_rating_file, index=False)

def print_rsvp_data(file_name):
    events = pd.read_csv(file_name)
    print(len(events))
    print(len(events['memberId'].unique()))
    print(len(events['eventId'].unique()))
    events = events.groupby('memberId').filter(lambda x : len(x) >= 5)
    print(len(events))
    print(len(events['memberId'].unique()))
    print(len(events['eventId'].unique()))

def main():
    print("Main method")
    #perform_train_test_split()
    generate_librec_rating_file()
    #print_rsvp_data(ny_file_name)

if __name__ == '__main__':
    main()
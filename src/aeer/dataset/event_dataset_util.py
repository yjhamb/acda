'''
Utility methods for the event dataset
'''

import pandas as pd

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

def print_rsvp_data(file_name):
    events = pd.read_csv(file_name)
    print(events['rsvpResponse'].unique())
    print(events['rsvpRating'].unique())

def main():
    print("Main method")
    print_rsvp_data(sfo_file_name_new)

if __name__ == '__main__':
    main()
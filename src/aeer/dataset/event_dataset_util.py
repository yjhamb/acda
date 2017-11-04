'''
Utility methods for the event dataset
'''

import pandas as pd

ny_file_name = "../../../dataset/rsvp_ny.csv"
sfo_file_name = "../../../dataset/rsvp_sfo.csv"
dc_file_name = "../../../dataset/rsvp_dc.csv"
chicago_file_name = "../../../dataset/rsvp_chicago.csv"
chicago_file_name_new = "../../../dataset/rsvp_chicago_new.csv"

def update_rsvp_data(file_name):
    events = pd.read_csv(file_name)
    print(events['rsvpRating'].unique())
    events.loc[events['rsvpRating'] == -1, 'rsvpRating'] = 1
    print(events['rsvpResponse'].unique())
    #pd.DataFrame.to_csv(chicago_file_name_new)

def main():
    print("Main method")
    update_rsvp_data(chicago_file_name)

if __name__ == '__main__':
    main()
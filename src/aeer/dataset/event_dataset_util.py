'''
Utility methods for the event dataset
'''

import pandas as pd

ny_file_name = "../../../dataset/rsvp_ny.csv"
sfo_file_name = "../../../dataset/rsvp_sfo.csv"
dc_file_name = "../../../dataset/rsvp_dc.csv"
chicago_file_name = "../../../dataset/rsvp_chicago.csv"
chicago_file_name_new = "Users/yjhamb/Projects/aeer/dataset/rsvp_chicago_new.csv"

def update_rsvp_data(file_name):
    events = pd.read_csv(file_name, dtype = {'memberId': str,'memberLat': str,
                                            'memberLon': str,'eventId': str,
                                            'eventTime': str,'eventYesRSVPCount': str,
                                            'groupId': str,'venueId': str,
                                            'venueLat': str,'venueLon': str,
                                            'rsvpTime': str,'rsvpResponse': str,
                                            'rsvpRating': str,'eventDay': str,
                                            'eventPeriod': str})
    print(events.dtypes)
    print(events['rsvpRating'].unique())
    #events.loc[events['rsvpResponse'] == 'watching', 'rsvpRating'] = 1
    #events.loc[events['rsvpResponse'] == 'waitlist', 'rsvpRating'] = 1
    print(events['rsvpResponse'].unique())
    print(events['rsvpRating'].unique())
    pd.DataFrame.to_csv(chicago_file_name_new, index=False)

def main():
    print("Main method")
    update_rsvp_data(chicago_file_name)

if __name__ == '__main__':
    main()
'''
Functions that operate on the user-group association dataset
'''

import pandas as pd

ny_file_name = "../../../dataset/user_group_ny.csv"
sfo_file_name = "../../../dataset/user_group_sfo.csv"
dc_file_name = "../../../dataset/user_group_dc.csv"
chicago_file_name = "../../../dataset/user_group_chicago.csv"

class UserGroupData(object):

    
    def __init__(self, file_name):
        self.user_group_data = pd.read_csv(file_name)

        
    def get_user_groups(self, user_id):
        return self.user_group_data.groupId[self.user_group_data.userId == user_id].unique()

    
    def is_user_in_group(self, user_id, group_id):
        user_groups = self.user_group_data.groupId[self.user_group_data.userId == user_id].unique()
        if group_id in user_groups:
            return True
        else:
            return False    
        

def main():
    print("main method")
    
if __name__ == '__main__':
    main()
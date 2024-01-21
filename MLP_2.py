from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle # Standard python module to save things
import os
from FileOperations import FileOperations

"""
The class DataFrame Helper, a class to easily perform common df
actions like reading into a csv file, data aggregation, and cleansing.
"""
class DFHelp():
    # A 'Global' variable to share the df between classes
    shared_df = None

    def __init__(self, csv_file_name):
        print(csv_file_name)
        DFHelp.shared_df = self.read_csv_data(csv_file_name)
        self.featureClass = Feature_Engineering()


    def get_df(self):
        return DFHelp.shared_df


    def get_head(self, n=10):
        print(DFHelp.shared_df.head(n))


    def get_info(self):
        print("--------------- INFO ---------------")
        print(f"Columns: {DFHelp.shared_df.columns}")
        print(f"Info: {DFHelp.shared_df.info()}")
        print(f"DF Head: {DFHelp.shared_df.head(5)}")
        print(f"Number of NaN values: {DFHelp.shared_df.isnull().sum().sum()}")
        print("--------------- End ---------------")


    def read_csv_data(self, csv_file_name: str) -> pd.DataFrame:
        # Get the current path of the current running script
        # combines the name of the folder of the running script + relative location of our csv file, win and linux compatiable.
        script_path = os.path.abspath(__file__)
        path = os.path.join(os.path.dirname(script_path), csv_file_name)
        print(f"script_path: {script_path}, final_path: {path}")
        df = pd.read_csv(path)
        return df  


    def save_to_csv(self, csv_file_name: str):
        script_path = os.path.abspath(__file__)
        path = os.path.join(os.path.dirname(script_path), csv_file_name)
        DFHelp.shared_df.to_csv(path, index=False)


    def convert_to_datetime(self, date_col: str, remove_timeZone = False):
        #remove the the MST/MDT, Turn the dates into date objects and sort
        if remove_timeZone == True:
            DFHelp.shared_df[date_col] = DFHelp.shared_df[date_col].str[:-4]
        DFHelp.shared_df[date_col] = pd.to_datetime(DFHelp.shared_df[date_col], format="%m/%d/%Y %H:%M")
        DFHelp.shared_df.sort_values(by=date_col, inplace=True)


    def convert_hour_to_date(self, date_time_col):
        # in this format: %HH:MM, converts it to a date object in mountain daylight time
        for index, row in DFHelp.shared_df.iterrows():
            current_time = datetime.now()
            hr, min = DFHelp.shared_df.at[index, date_time_col].split(":")
            mtn_time_entry = current_time.replace(hour=int(hr), minute=int(min), second=0, microsecond=0)
            DFHelp.shared_df.at[index, date_time_col] = mtn_time_entry


    def convert_to_float16(self, *columns):
        for column in columns:
            DFHelp.shared_df[column] = DFHelp.shared_df[column].astype('float16')


    # def fill_NAN_with_mean(self, column: str):
        # DFHelp.shared_df[column].fillna(DFHelp.shared_df[column], method="ffill", inplace=True)


    def remove_NAN(self):
        DFHelp.shared_df = DFHelp.shared_df.dropna()

    def zero_forwardFill(self):
        DFHelp.shared_df.fillna(method="ffill", inplace=True)
        DFHelp.shared_df.fillna(0, inplace=True)

    def drop_columns(self, *columns):
        for column in columns:
            DFHelp.shared_df = DFHelp.shared_df.drop([column], axis="columns")


    def keep_columns(self, columns: list):
        print(f"Columns Not Used: ")
        print(type(columns))
        for column in DFHelp.shared_df.columns:
            if column not in columns and column != "Date_Time":
                print(f'    -{column}')
                DFHelp.shared_df = DFHelp.shared_df.drop([column], axis="columns")
        print(f"Columns Used: {DFHelp.shared_df.columns}")


    def rename_columns(self, old_col, new_col):
        print(f"columns: {DFHelp.shared_df.columns}")
        new_col_names = {key: value for key, value in zip(old_col, new_col)}
        DFHelp.shared_df.rename(columns=new_col_names, inplace=True)


    def cardinal_to_degrees(self, degrees_col_name):
        """
        Converts cardinal directions to degrees
        """
        d={'N':0, 'NNE':22.5,"NE":45,"ENE":67.5, 'E':90,'ESE':112.5, 'SE':135,'SSE':157.5,
            'S':180,'SSW':202.5, 'SW':225,'WSW':247.5, 'W':270,'WNW':292.5,'NW':315,
            'NNW':337.5, 'N':0,'North':0,'East':90,'West':270,'South':180, 'A':0}
        DFHelp.shared_df[degrees_col_name]=DFHelp.shared_df[degrees_col_name].str.strip().replace(d)


"""
A class component of DFHelp assists with creating 
various features for the df
"""
class Feature_Engineering():

    def aggregate_by_hour(self, columns_to_mean, hours="H"):
        # Combines every row by a specified number of hours. Combines the data by the mean
        # hours should be in this format: "nH" where n is a positive real number
        # set the index as the date_time column and remove datetime to avoid errors
        DFHelp.shared_df.set_index('Date_Time', inplace=True)
        columns_to_mean.remove("Date_Time")
        DFHelp.shared_df = DFHelp.shared_df[columns_to_mean].resample(hours).mean()
        DFHelp.shared_df = DFHelp.shared_df.reset_index()


    def z_score_normalize(self, columns: list, *columns_to_exclude):
        for column in columns:
            if column not in columns_to_exclude:
                DFHelp.shared_df[column] = (DFHelp.shared_df[column] - DFHelp.shared_df[column].mean()) / DFHelp.shared_df[column].std()


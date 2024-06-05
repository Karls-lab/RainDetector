import MLP_2
from TransformerModel import TransformerModel
import os
import pandas as pd


def set_up_training_data(file_name, features) -> pd.DataFrame:
    """
    This function sets up the training data, cleans the data, and adds new feature columns 
    using the helper methods in MLP_2.py
    """
    df_class = MLP_2.DFHelp(os.path.join('weather_data', file_name))
    df_class.keep_columns(features)
    df_class.get_info()
    df_class.convert_to_datetime("Date_Time", remove_timeZone=True)
    df_class.featureClass.aggregate_by_hour(features, hours="3H") # 8 total readings per day
    # normalize all columns except for datetime and wind_direction
    df_class.featureClass.z_score_normalize(features, "Date_Time", "wind_direction")
    df_class.zero_forwardFill() # forward fill, then fill with 0's
    df_class.convert_to_float16(features) # For memory efficiency

    print("Finished Setting up Training Data. Here is the head:")
    df_class.get_head(20)
    return df_class.get_df()


def run(data_file_name: str, model_save_folder, features: list, model="LSTM"):
    data = set_up_training_data(data_file_name, features)
    data.head().to_csv("training_sample.csv")
    if model == "Transformer":
        print("Transformer Model")
        transModel = TransformerModel(df=data, features=features, predictions="air_temp")
        transModel.split_into_sequences()
        transModel.createTransformerModel()
        transModel.trainModel()
        transModel.saveModel()
    # create_weekly_model(df=data, features=features, save_path=model_save_folder)

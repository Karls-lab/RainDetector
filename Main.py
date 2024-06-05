import ModelCreator
# import CurrentWeather


def main():
    year_file = "TooeleWeatherData1yr.csv"
    small_test = "small_test.csv"
    bolinder5yr = "BolinderAirport5yr.csv"
    monthData = "BolinderJan2018.csv"

    # Features for Bolinder Airport 
    # Need to add wind direction
    features = ['altimeter', 'air_temp', 'dew_point_temperature', 'relative_humidity', 
                    'wind_speed', 'wind_gust', 'wind_direction', 'Date_Time']
    
    ModelCreator.run(bolinder5yr, model_save_folder="saved_Models", features=features, model="Transformer")
    
    # url = "https://mesowest.utah.edu/cgi-bin/droman/meso_base_dyn.cgi?stn=G2864"
    # CurrentWeather.run(url, features=features)

    # Run the model on weather data 
    # clean the data first
    # df = ModelCreator.set_up_training_data("Bolinder2010.csv", features)





if __name__ == "__main__":
    main()
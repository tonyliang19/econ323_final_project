# helper to get data
import pandas as pd
import os
import sys
from urllib.request import urlretrieve

# get data, downloads from url if not in relative path
def get_data(data_path):
    # checkes if file exists, if not downloaded to that path
    url = "https://raw.githubusercontent.com/tonyliang19/econ323_final_project/main/data/boston_housing_data.csv"
    if not os.path.isfile(data_path):
        print(f"You don't have the file yet, and it will be downloaded to: {os.path.abspath(data_path)}")
        print("Downloading now, wait a few secs")
        urlretrieve(url, data_path)
        print("Done!")
    else:
        pass
        
    data = pd.read_csv(data_path)
    return data
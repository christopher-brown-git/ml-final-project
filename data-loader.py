import pandas as pd
import numpy as np
import os

def load_data():
    paths_to_files = []
    directory = "data"

    for name in os.scandir(directory):
        if name.is_dir():
            for obj in os.scandir(name):
                if obj.is_file():
                    paths_to_files.append(obj.path)
    
    file_objects = []


    data = pd.read_csv(paths_to_files[0])
    
    print("test1")
    data = pd.get_dummies(data)
    print("test2")
    
    for col in data.columns:
        print(col)

load_data()
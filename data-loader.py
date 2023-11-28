import pandas as pd
import numpy as np
import os

def load_data():
    paths_to_files = []
    directory = "data"

    card_list_path = ""

    for name in os.scandir(directory):
        if name.is_dir():
            for obj in os.scandir(name):
                if obj.is_file():
                    paths_to_files.append(obj.path)
        elif name.is_file and name.path.split('/')[1] == 'CardMasterListSeason18_12082020.csv':
            card_list_path = name.path

    #read in dataframe
    df = pd.read_csv(paths_to_files[0])

    card_list_file = open(card_list_path, 'r')

    #dictionary mapping from card codes to card names
    card_dict = {}

    #set of columns not to drop
    keep_cols = set()

    #create dummy variables
    index_w = 0
    for line in card_list_file:
        arr = line.split(',')
        card_code = arr[0]
        card_name = arr[1].split('\n')[0]

        card_col_w = card_name + '_w'
        card_col_l = card_name + '_l'

        #create dummy variable for if the winner had a certain card
        #and for if the loser had a certain card
        df.insert(index_w, card_col_w, '')
        df.insert(index_w*2 + 1, card_col_l, '')

        keep_cols.add(card_col_w)
        keep_cols.add(card_col_l)
        
        index_w += 1

        card_dict[card_code] = card_name

    #create outcome column, it will always be ones because the winning deck
    #is the deck with lower column indices

    df['Y'] = 1
    keep_cols.add('Y')

    card_list_file.close()

    # print(keep_cols)

    #remove unnecessary columns
    drop_cols = []
    for col in df.columns:
        if col not in keep_cols:
            drop_cols.append(col)
    
    #print(drop_cols)
    df = df.drop(columns=drop_cols)


    # file_objects = []
    
    # print("test1")
    # data = pd.get_dummies(data)
    # print("test2")
    
    #print("00000000000000000000000000000")

    for col in df.columns:
        print(col)

load_data()
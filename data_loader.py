#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.model_selection import train_test_split

def create_data():
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

    #NEW APPROACH
    #drop unneccessary columns
    #create list of dictionaries and then create new dataframe from this 
    #save resulting dataframe into a csv file


    #dictionary mapping from card codes to card names
    card_dict = {}

    #open csv containing the mapping between card codes and card names
    card_list_file = open(card_list_path, 'r')

    #create mapping from card codes to card names
    for num, line in enumerate(card_list_file):
        #skip first line of file
        if num == 0:
            continue

        arr = line.split(',')
        card_code = arr[0]
        card_name = arr[1].split('\n')[0]
        
        card_dict[card_code] = card_name

    card_list_file.close()

    #read in dataframe
    df = pd.read_csv(paths_to_files[0])


    cols_to_keep = {"winner.cards.list", "loser.cards.list"}

    cols_to_drop = []
    for col in df.columns:
        if col not in cols_to_keep:
            cols_to_drop.append(col)
    
    df = df.drop(cols_to_drop, axis=1)

    new_rows = []

    #player 1's cards are always the lower indexed columns
    #player 2's cards are always the higher indexed columns
    #outcome vector is 1 if player1 won and 2 if player2 won

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        new_row_1 = {}    
        new_row_2 = {}    

        cards_in_row = set()
        for col in cols_to_keep:
            player = col.split('.')[0][0]

            codes_as_str = row[col][1:-1]
            codes = codes_as_str.split(', ')

            if player == "w":
                for code in codes:
                    cards_in_row.add(card_dict[str(code)] + '_w')
            else:
                for code in codes:
                    cards_in_row.add(card_dict[str(code)] + '_l')
        
        #flip coin to determine whether or not player1 is the winner or player2 is the winner
        # > .5 means that player1 is the winner
        # <= .5 means player2 is the winner

        Y = "Y"
        winner_is_1 = 0
        new_row_1[Y] = 0
        if np.random.rand() > 0.5:
            winner_is_1 = 1
            new_row_1[Y] = 1


        for card in card_dict.values():
            card_w = card + '_w'
            card_l = card + '_l'

            card_1 = card + '_1'
            card_2 = card + '_2'

            if card_w in cards_in_row:
                if winner_is_1:
                    new_row_1[card_1] = 1
                else:
                    new_row_2[card_2] = 1
            else:
                if winner_is_1:
                    new_row_1[card_1] = 0
                else:
                    new_row_2[card_2] = 0

            
            if card_l in cards_in_row:
                if winner_is_1:
                    new_row_2[card_2] = 1
                else:
                    new_row_1[card_1] = 1
            else:
                if winner_is_1:
                    new_row_2[card_2] = 0
                else:
                    new_row_1[card_1] = 0

        new_row_1.update(new_row_2)
        new_rows.append(new_row_1)

        #should be plenty of rows
        if index > 1500000:
            break
    
    df_small = pd.DataFrame(new_rows)
        
    df_small.to_csv("df_small_1.csv")


def load_data():
    #load in data with pandas
    data = pd.read_csv("df_small_1.csv")

    #get outcome vector (it's all 1's bc the player with cards
    #in lower indexed columns is the winner)
    Y = np.array([x for x in data["Y"]])

    #create feature matrix
    feature_names = [col for col in data.columns]
    data_features = data[feature_names]
    Xmat = data_features.to_numpy(dtype=np.int8)

    #split feature matrix into training, validation, and testing splits
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=1)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.2, random_state=1)

    #data is just 1-hot vectors so no need to standardize the data

    #add a column of ones for the intercept term because we are doing logistic regression
    Xmat_train = np.column_stack((np.ones(len(Xmat_train)), Xmat_train))
    Xmat_val = np.column_stack((np.ones(len(Xmat_val)), Xmat_val))
    Xmat_test = np.column_stack((np.ones(len(Xmat_test)), Xmat_test))
    feature_names = ["intercept"] + feature_names

    #return training, validation, and testing datasets
    return feature_names, {"Xmat_train": Xmat_train, "Xmat_val": Xmat_val, "Xmat_test": Xmat_test,
                           "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test}

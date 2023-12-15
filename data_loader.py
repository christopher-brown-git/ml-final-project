#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import pickle
import tqdm
import sys
from os.path import exists
from sklearn.model_selection import train_test_split
sys.path.insert(1, 'webscraper/venv/')

#creats a new dataframe with the only features being the cards used in each deck
def create_data_simple(rows_of_data, data_path):
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

    #APPROACH
    #drop unneccessary columns
    #create list of dictionaries and then create new dataframe from this 
    #save resulting dataframe into a csv file


    #dictionary mapping from card code -> (card name, elixir)
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
        if index >= rows_of_data:
            break
    
    df_small = pd.DataFrame(new_rows)
        
    df_small.to_csv(data_path)

#loads the dataframe with the only features being the cards used in each deck
def load_data_simple(data_path):
    #load in data with pandas
    data = pd.read_csv(data_path)

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


#***************************START COMPLEX*************************************

def create_data_complex(data_path):
    if not exists(data_path):
        games_to_df()
    
#load in complex data; there are continuous variables so need to standardize the continuous variables
def load_data_complex(data_path):
    #load in data with pandas
    data = pd.read_csv(data_path)

    continuous_features = {
        'elixir_cost_1', 'common_count_1', 'rare_count_1', 'epic_count_1', 'legendary_count_1', 
        'champion_count_1', 'spell_count_1', 'structure_count_1', 'spawner_count_1', 'melee_short_count_1', 
        'melee_medium_count_1', 'melee_long_count_1', 'ranged_count_1', 'air_count_1', 'ground_count_1', 
        'defensive_building_count_1', 'siege_building_count_1', 'total_card_level_count_1', 'troop_count_1', 
        'Tot_Healing_1', 'Tot_Hitpoints_1', 'Tot_Towerdamage_1', 'Tot_Deathdamage_1', 'Tot_Damage_1', 'Tot_Shield_1', 
        'Tot_Buildingdamage_1', 'Avg_Hitspeed_1', 'Avg_Spellradius_1', 'Avg_Range_1', 'Avg_Speed_1',

        'elixir_cost_2', 'common_count_2', 'rare_count_2', 'epic_count_2', 'legendary_count_2', 
        'champion_count_2', 'spell_count_2', 'structure_count_2', 'spawner_count_2', 'melee_short_count_2', 
        'melee_medium_count_2', 'melee_long_count_2', 'ranged_count_2', 'air_count_2', 'ground_count_2', 
        'defensive_building_count_2', 'siege_building_count_2', 'total_card_level_count_2', 'troop_count_2', 
        'Tot_Healing_2', 'Tot_Hitpoints_2', 'Tot_Towerdamage_2', 'Tot_Deathdamage_2', 'Tot_Damage_2', 'Tot_Shield_2', 
        'Tot_Buildingdamage_2', 'Avg_Hitspeed_2', 'Avg_Spellradius_2', 'Avg_Range_2', 'Avg_Speed_2'    
    }

    #get outcome vector (it's all 1's bc the player with cards
    #in lower indexed columns is the winner)
    Y = np.array([x for x in data["Y"]])

    # standardize the data
    for col in data.columns:
        if col in continuous_features:
            data[col] = (data[col]-data[col].mean())/data[col].std()

    #create feature matrix
    feature_names = [col for col in data.columns]   
    data_features = data[feature_names]
    Xmat = data_features.to_numpy(dtype=np.int8)

    #split feature matrix into training, validation, and testing splits
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=1)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.2, random_state=1)

    #need to standardize continous variables

    
    #add a column of ones for the intercept term because we are doing logistic regression
    Xmat_train = np.column_stack((np.ones(len(Xmat_train)), Xmat_train))
    Xmat_val = np.column_stack((np.ones(len(Xmat_val)), Xmat_val))
    Xmat_test = np.column_stack((np.ones(len(Xmat_test)), Xmat_test))
    feature_names = ["intercept"] + feature_names

    #return training, validation, and testing datasets
    return feature_names, {"Xmat_train": Xmat_train, "Xmat_val": Xmat_val, "Xmat_test": Xmat_test,
                           "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test}


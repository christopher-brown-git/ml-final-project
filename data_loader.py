import pandas as pd
import numpy as np
import os
import tqdm

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


    cols_to_keep = {"winner.card1.id", "winner.card2.id",
                        "winner.card3.id", 	"winner.card4.id",
                        "winner.card5.id",	"winner.card6.id",
                        "winner.card7.id",	"winner.card8.id", 
                        "loser.card1.id", "loser.card2.id",
                        "loser.card3.id", 	"loser.card4.id",
                        "loser.card5.id",	"loser.card6.id",
                        "loser.card7.id",	"loser.card8.id"}

    cols_to_drop = []
    for col in df.columns:
        if col not in cols_to_keep:
            cols_to_drop.append(col)
    
    df = df.drop(cols_to_drop, axis=1)

    new_rows = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        new_row_w = {}    
        new_row_l = {}    

        cards_in_row = set()
        for col in cols_to_keep:
            card_code = str(row[col])

            player = col.split('.')[0][0]

            if player == "w":
                cards_in_row.add(card_dict[card_code] + '_w')
            else:
                cards_in_row.add(card_dict[card_code] + '_l')
            
        for card in card_dict.values():
            card_w = card + '_w'
            if card_w in cards_in_row:
                new_row_w[card_w] = 1
            else:
                new_row_w[card_w] = 0
        
            card_l = card + '_l'
            if card_l in cards_in_row:
                new_row_l[card_l] = 1
            else:
                new_row_l[card_l] = 0
        
        new_row_w.update(new_row_l)
        new_rows.append(new_row_w)

        #should be plenty of rows
        if index > 1500000:
            break
    
    df_small = pd.DataFrame(new_rows)
        
    df_small.to_csv("df_small_1.csv")


load_data()
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


    #create empty dataframe
    df_small = pd.DataFrame()

    #dictionary mapping from card codes to card names
    card_dict = {}

    #open csv containing the mapping between card codes and card names
    card_list_file = open(card_list_path, 'r')

    #add new columns to new dataframe for the categorical variables indicating whether or not
    #the winners and losers' decks contain certain cards AND create dict mapping card codes to card names
    index_w = 0
    for num, line in enumerate(card_list_file):
        #skip first line of file
        if num == 0:
            continue

        arr = line.split(',')
        card_code = arr[0]
        card_name = arr[1].split('\n')[0]

        card_col_w = card_name + '_w'
        card_col_l = card_name + '_l'

        #create dummy variable for if the winner had a certain card
        #and for if the loser had a certain card
        df_small.insert(index_w, card_col_w, '')
        df_small.insert(index_w*2 + 1, card_col_l, '')

        index_w += 1
        
        card_dict[card_code] = card_name

    card_list_file.close()

    #create outcome column, it will always be ones because the winning deck
    #is the deck with lower column indices
    df_small.insert(0, 'Y', 1)

    #read in dataframe
    df = pd.read_csv(paths_to_files[0])


    cols_to_transfer = ["winner.card1.id", "winner.card2.id",
                        "winner.card3.id", 	"winner.card4.id",
                        "winner.card5.id",	"winner.card6.id",
                        "winner.card7.id",	"winner.card8.id", 
                        "loser.card1.id", "loser.card2.id",
                        "loser.card3.id", 	"loser.card4.id",
                        "loser.card5.id",	"loser.card6.id",
                        "loser.card7.id",	"loser.card8.id",]

    #assign categorical variables in df_small based on values in df
    for index, row in df.iterrows():
        for col in df_small.col


load_data()
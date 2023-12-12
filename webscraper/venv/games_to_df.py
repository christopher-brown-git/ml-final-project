import pandas as pd
from os.path import exists
import pickle
import tqdm
import game_scraper

path = "/home/scratch/24cjb4/games.pkl"

def convert():
    if not exists(path):
        game_scraper()
    
    #open pickle file
    games_dict = ''
    with open(path, 'rb') as file:
        games_dict = pickle.load(file)

    for (key, val) in games_dict.items():
        

if __name__ == "__main__":
    convert()
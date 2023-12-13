import pandas as pd
from os.path import exists
import pickle
import tqdm
import game_scraper

path = "/home/scratch/24cjb4/games.pkl"

cols_to_keep = ["winner.cards.list", "winner.totalcard.level", "winner.troop.count", 
                "winner.structure.count", "winner.spell.count", "winner.common.count", 
                "winner.rare.count", "winner.epic.count", "winner.legendary.count", "winner.elixir.average",
                "loser.cards.list", "loser.totalcard.level", "loser.troop.count", 
                "loser.structure.count", "loser.spell.count", "loser.common.count", 
                "loser.rare.count", "loser.epic.count", "loser.legendary.count", "loser.elixir.average"]

new_features_map = {"Healing": "Tot_Healing", "Hitpoints" : "Tot_Hitpoints", "Tower damage": "Tot_Towerdamage",
                        "Hitspeed": "Avg_Hitspeed", "Death damage": "Tot_Deathdamage", "Damage": "Tot_Damage",
                        "Shield": "Tot_Shield", "Spell radius": "Avg_Spellradius", "Range": "Avg_Range",
                        "Speed": "Avg_Speed", "Building damage": "Tot_Buildingdamage", "Siege Building": "Tot_Siege_Buildings",
                        "Defensive Building":"Tot_Defensive_Buildings", "Melee Range": "Avg_MeleeRange", "Air":"Tot_Air", "Ground": "Tot_Ground"}

card_features = ['Goblins', 'FireSpirit', 'Bats', 'Skeletons', 'Barbarians', 'SpearGoblins', 'RoyalRecruits']

spawners = {"Tombstone", "Furnace", "Barbarian Hut", "Goblin Hut", "Goblin Drill", "Graveyard"}

dps = {"Tornado", "Poison", "Earthquake"}

def convert():
    if not exists(path):
        game_scraper()
    
    #open pickle file
    games_dict = ''
    with open(path, 'rb') as file:
        games_dict = pickle.load(file)

    #load in clash royale card data
    stats = ''
    with open('stats.pkl', 'rb') as file:
        stats = pickle.load(file)
    
    new_rows = []

    for (key, val) in games_dict.items():
        new_row = {}



if __name__ == "__main__":
    convert()
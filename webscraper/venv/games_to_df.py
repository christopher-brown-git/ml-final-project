import pandas as pd
from os.path import exists
import pickle
import tqdm
import game_scraper
import numpy as np

path = "/home/scratch/24cjb4/games.pkl"

champions = ("champion_count", {"Little Prince", "Golden Knight", "Skeleton King", "Mighty Miner", "Archer Queen", "Monk"})

legendary = ("legendary_count", {"The Log", "Miner", "Princess", "Ice Wizard", "Royal Ghost", "Bandit", "Fisherman",
             "Electro Wizard", "Inferno Dragon", "Phoenix", "Magic Archer", "Lumberjack", "Night Witch", 
             "Mother Witch", "Ram Rider", "Graveyard", "Sparky", "Mega Knight", "Lava Hound"})

epic = ("epic_count", {"Mirror", "Barbarian Barrel", "Wall Breakers", "Rage", "Goblin Barrel", "Guards", "Skeleton Army", 
        "Clone", "Tornado", "Baby Dragon", "Dark Prince", "Freeze", "Poison", "Hunter", "Goblin Drill", 
        "Witch", "Balloon", "Prince", "Electro Dragon", "Bowler", "Executioner", "Cannon Cart", 
        "Giant Skeleton", "Lightning", "Goblin Giant", "X-Bow", "P.E.K.K.A", "Electro Giant", "Golem"})

rare = ("rare_count", {"Heal Spirit", "Ice Golem", "Tombstone", "Mega Minion", "Dart Goblin", "Earthquake", "Elixir Golem", 
        "Fireball", "Mini P.E.K.K.A", "Musketeer", "Goblin Cage", "Valkyrie", "Battle Ram", "Bomb Tower", 
        "Flying Machine", "Hog Rider", "Battle Healer", "Furnace", "Zappies", "Giant", "Goblin Hut", 
        "Inferno Tower", "Wizard", "Royal Hogs", "Rocket", "Barbarian Hut", "Elixir Collector", "Three Musketeers"})

common = ("common_count", {"Skeletons", "Electro Spirit", "Fire Spirit", "Ice Spirit", "Goblins", "Spear Goblins", 
        "Bomber", "Bats", "Zap", "Giant Snowball", "Archers", "Arrows", "Knight", "Minions", 
        "Cannon", "Goblin Gang", "Skeleton Barrel", "Firecracker", "Royal Delivery", "Skeleton Dragons", 
        "Mortar", "Tesla", "Barbarians", "Minion Horde", "Rascals", "Royal Giant", "Elite Barbarians", 
        "Royal Recruits", "Warmth"})

spells = ("spell_count", {"Zap", "Giant Snowball", "Arrows", "Royal Delivery", "Earthquake", "Fireball", 
          "Rocket", "Mirror", "Barbarian Barrel", "Rage", "Goblin Barrel", "Clone", 
          "Tornado", "Freeze", "Poison", "Lightning", "The Log", "Graveyard"})

structures = ("structure_count", {"Cannon", "Mortar", "Tesla", "Tombstone", "Goblin Cage", "Bomb Tower", 
              "Furnace", "Goblin Hut", "Inferno Tower", "Barbarian Hut", "Elixir Collector", 
              "Goblin Drill", "X-Bow"})

spawners = ("spawner_count", {"Tombstone", "Furnace", "Barbarian Hut", "Goblin Hut", "Goblin Drill", "Graveyard"})

#EXTRA CATEGORIES
melee_short = ("melee_short_count", {"Skeletons", "Goblins", "Goblin Gang", "Barbarians", "Rascals", "Elixir Golem", "Mini P.E.K.K.A", 
                    "Goblin Cage", "Barbarian Barrel", "Goblin Barrel", "Skeleton Army", "Giant Skeleton", 
                    "Bandit", "Graveyard", "Lumberjack"})

melee_medium = ("melee_medium_count", {"Bats", "Knight", "Elite Barbarians", "Valkyrie", "Dark Prince", "Goblin Giant", 
                    "P.E.K.K.A", "Electro Giant", "Miner", "Royal Ghost", "Fisherman", "Mega Knight", "Golden Knight", 
                    "Skeleton King", "Monk"})

melee_long = ("melee_long_count", {"Minions", "Royal Delivery", "Minion Horde", "Royal Recruits", "Mega Minion", "Battle Healer", "Guards", 
                    "Prince", "Phoenix", "Night Witch", "Mighty Miner"})

ranged = ("ranged_count", {"Spear Goblins", "Bomber", "Archers", "Cannon", "Goblin Gang", "Firecracker", 
          "Skeleton Dragons", "Rascals", "Dart Goblin", "Musketeer", "Flying Machine", 
          "Zappies", "Goblin Hut", "Wizard", "Three Musketeers", "Baby Dragon", "Hunter", 
          "Witch", "Electro Dragon", "Bowler", "Executioner", "Cannon Cart", "Goblin Giant", 
          "X-Bow", "Princess", "Ice Wizard", "Fisherman", "Electro Wizard", "Inferno Dragon", 
          "Magic Archer", "Mother Witch", "Ram Rider", "Sparky", "Little Prince", "Archer Queen"})

air = ("air_count", {"Bats", "Minions", "Skeleton Barrel", "Skeleton Dragons", "Minion Horde", "Mega Minion", "Flying Machine", 
            "Baby Dragon", "Balloon", "Electro Dragon", "Inferno Dragon", "Phoenix", "Lava Hound"})

ground = ("ground_count", {"Skeletons", "Electro Spirit", "Fire Spirit", "Ice Spirit", "Goblins", "Spear Goblins",
            "Bomber", "Archers", "Knight", "Goblin Gang", "Firecracker", "Royal Delivery", "Barbarians", "Rascals", "Royal Giant",
            "Elite Barbarians", "Royal Recruits", "Heal Spirit", "Ice Golem", "Dart Goblin", "Elixir Golem", "Mini P.E.K.K.A", "Musketeer",
            "Goblin Cage", "Valkyrie", "Battle Ram", "Hog Rider", "Battle Healer", "Zappies", "Giant", "Wizard", "Royal Hogs",
            "Three Musketeers", "Barbarian Barrel", "Wall Breakers", "Goblin Barrel", "Guards", "Skeleton Army", "Dark Prince",
            "Hunter", "Witch", "Prince", "Bowler", "Executioner", "Cannon Cart", "Giant Skeleton", "Goblin Giant",
            "P.E.K.K.A", "Electro Giant", "Golem", "Miner", "Princess", "Ice Wizard", "Royal Ghost", "Bandit", "Fisherman",
            "Electro Wizard", "Magic Archer", "Lumberjack", "Night Witch", "Mother Witch", "Ram Rider", "Sparky",
            "Mega Knight", "Little Prince", "Golden Knight", "Skeleton King", "Mighty Miner", "Archer Queen", "Monk"})

defensive_buildings = ("defensive_building_count", {"Cannon", "Tesla", "Bomb Tower", "Inferno Tower"})

siege_buildings = ("siege_building_count", {"Mortar", "X-Bow"})

cols_to_keep = ["winner.cards.list", "winner.totalcard.level", "winner.troop.count", 
                "winner.structure.count", "winner.spell.count", "winner.common.count", 
                "winner.rare.count", "winner.epic.count", "winner.legendary.count", "winner.elixir.average",
                "loser.cards.list", "loser.totalcard.level", "loser.troop.count", 
                "loser.structure.count", "loser.spell.count", "loser.common.count", 
                "loser.rare.count", "loser.epic.count", "loser.legendary.count", "loser.elixir.average"]

card_features = ['Goblins', 'FireSpirit', 'Bats', 'Skeletons', 'Barbarians', 'SpearGoblins', 'RoyalRecruits']

dps = {"Tornado", "Poison", "Earthquake"}

other_features_map = {"Healing": "Tot_Healing", "Hitpoints" : "Tot_Hitpoints", "Tower damage": "Tot_Towerdamage",
                    "Death damage": "Tot_Deathdamage", "Damage": "Tot_Damage", "Shield": "Tot_Shield", 
                    "Building damage": "Tot_Buildingdamage", "Hitspeed": "Avg_Hitspeed", "Spell radius": "Avg_Spellradius", 
                    "Range": "Avg_Range", "Speed": "Avg_Speed"}

def initialize(row, player, info, card_dict):
    cards_in_hand = {}

    #card_dict: id -> (name, elixirCost)
    for (id, level) in info[player]:
        card_name = card_dict[id][0] #FIX ERRORS LIKE THESE
        cards_in_hand[card_name] = int(level)

    row['elixir_cost'] = 0

    row['common_count'] = 0
    row['rare_count'] = 0
    row['epic_count'] = 0
    row['legendary_count'] = 0
    row['champion_count'] = 0
    row['spell_count'] = 0
    row['structure_count'] = 0
    row['spawner_count'] = 0
    row['melee_short_count'] = 0
    row['melee_medium_count'] = 0
    row['melee_long_count'] = 0
    row['ranged_count'] = 0
    row['air_count'] = 0
    row['ground_count'] = 0
    row['defensive_building_count'] = 0
    row['siege_building_count'] = 0
    row['total_card_level_count'] = 0
    row['troop_count'] = 0

    for f in other_features_map.values():
        #if we average it, we need the feature in a list
        #if we sum it, the feature can remain a value
        if f[0] == "T":
            row[f] = 0
        else:
            row[f] = []

    return cards_in_hand

def create_features(stats, card_dict, card_name, elixir_cost, row, cards_in_hand):

    #ignore mirror for now
    if card_name == "Mirror":
        if card_name in cards_in_hand.keys():
            row[card_name] = 1
        else:
            row[card_name] = 0

        return

    #ignore warmth except for elixir count
    if card_name == "Warmth":
        if card_name in cards_in_hand.keys():
            row[card_name] = 1
            row['elixir_cost'] += elixir_cost
        else:
            row[card_name] = 0
          
        return
    
    #ME/ROW 1

    if card_name in cards_in_hand.keys():
        level = cards_in_hand[card_name]
        row[card_name] = 1
        row['elixir_cost'] += elixir_cost
        
        #handle counts
        for category in [common, rare, epic, legendary, champions, spells, 
                        structures, spawners, melee_short, melee_medium, melee_long, ranged, air, ground,
                        defensive_buildings, siege_buildings]:
            
            if card_name in category[1]:
                row[category[0]] += 1
            
        if card_name not in structures and card_name not in spells:
            row["troop_count"] += 1
        
        row["total_card_level_count"] += level

        #go through other features:
        #if one of these features is a list, it should be accumulated
        #if it is a value, it should be averaged

        for feature in other_features_map.keys():
            actual_feature = other_features_map[feature]
            if feature in stats[card_name].keys():
                mult = int(stats[card_name]['mult'])

                if isinstance(stats[card_name][feature], list):
                    #index into it
                    index = level - 1
                    value = int(stats[card_name][feature][index])*mult

                    if isinstance(row[actual_feature], list):
                        row[actual_feature].append(value)
                    else:
                        row[actual_feature] += value
                else:
                    #do not index into it
                    value = int(stats[card_name][feature])*mult

                    if isinstance(row[actual_feature], list):
                        row[actual_feature].append(value)
                    else:
                        row[actual_feature] += value

        #special case for spawners: just add the stats for all the troops the produce over their lifetime
        if card_name in spawners:
            d = stats[card_name]
            lifetime = int(d['Lifetime'])
            spawnspeed = int(d['SpawnSpeed'])
            spawnnumber = int(d['SpawnNumber'])
            spawnondeath = int(d['SpawnOnDeath'])

            num = int((lifetime//spawnspeed) * spawnnumber + spawnondeath)

            for _ in range(0, num):
                spawn = stats[card_name]['Spawn']
                for feature in stats[spawn]:
                    #do not consider categorical features here
                    if feature in other_features_map.values():
                        mult = int(stats[card_name]['mult'])

                        if isinstance(stats[card_name][feature], list):
                            #all spawners produce cards that are the same level as they are themselves
                            index = level - 1
                            value = int(stats[card_name][feature][index])*mult

                            if isinstance(row[feature], list):
                                row[feature].append(value)
                            else:
                                row[feature] += value
                        else:
                            value = int(stats[card_name][feature])*mult

                            if isinstance(row[feature], list):
                                row[feature].append(value)
                            else:
                                row[feature] += value
        
        #last special case
        if card_name == "Goblin Gang":
                #go through features that accumulate (total)
            for feature in stats["Goblins"].keys():
                #do not consider categorical features here
                if feature in other_features_map.values():
                    mult = int(stats[card_name]['mult'])

                    if isinstance(stats[card_name][feature], list):
                        index = level - 1
                        value = int(stats[card_name][feature][index])*mult

                        if isinstance(row[feature], list):
                            row[feature].append(value)
                        else:
                            row[feature] += value
                    else:
                        value = int(stats[card_name][feature])*mult

                        if isinstance(row[feature], list):
                            row[feature].append(value)
                        else:
                            row[feature] += value
                        
                        
            for feature in stats["Spear Goblins"].keys():
                #do not consider categorical features here
                if feature in other_features_map.values():
                    mult = int(stats[card_name]['mult'])

                    if isinstance(stats[card_name][feature], list):
                        #all spawners produce cards that are the same level as they are themselves
                        index = level - 1
                        value = int(stats[card_name][feature][index])*mult
                        
                        if isinstance(row[feature], list):
                            row[feature].append(value)
                        else:
                            row[feature] += value
                        
                    else:
                        value = int(stats[card_name][feature])*mult
                        
                        if isinstance(row[feature], list):
                            row[feature].append(value)
                        else:
                            row[feature] += value
    else:
        row[card_name] = 0

def convert(data_path):
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
    
    #load in dictionary from card code --> (card_name, elixirCost)
    card_dict = ''
    with open('card_dict.pkl', 'rb') as file:
        card_dict = pickle.load(file)
    
    new_rows = []

    i = 0
    for (key, info) in tqdm.tqdm(games_dict.items(), total=len(games_dict)):
        #until the end the first few columns is me, and the last few columns is opponent
        #decide order at end
        #outcome vector is 1 if the player in the first few columns beat the player in the last few columns

        new_row_me = {}
        new_row_opp = {}

        cards_in_my_hand = initialize(new_row_me, "me", info, card_dict)
        cards_in_opp_hand = initialize(new_row_opp, "opp", info, card_dict)

        for (card_id, (card_name, elixir_cost)) in card_dict.items():
            create_features(stats, card_dict, card_name, elixir_cost, new_row_me, cards_in_my_hand)
            create_features(stats, card_dict, card_name, elixir_cost, new_row_opp, cards_in_opp_hand)


        #average out features that are lists

        for row in [new_row_me, new_row_opp]:
            for feature in row.keys():
                if isinstance(row[feature], list):
                    if len(row[feature]) == 0:
                        row[feature] = 0
                    else:
                        row[feature] = sum(row[feature])//len(row[feature]) #for numerical reasons, should be fine
        
        me_is_1 = 0
        if np.random.rand() > 0.5:
            me_is_1 = 1

        #randomly assign "me" to be 1 with 50% probability because the "me" in the data is a 
        #player in a top ranked clan; since these players are better than average, if they are always
        #given the tag 1 the model may just predict that the player with tag 1 wins, naively

        #the outcome vector is 1 if player 1 wins and 0 if player 1 loses
        actual_row = {}

        if me_is_1:

            for (key, value) in new_row_me.items():
                actual_row[key + "_1"] = value

            for (key, value) in new_row_opp.items():
                actual_row[key + "_2"] = value
            
            if info["winner"] == 1:
                #me won and me is 1
                actual_row["Y"] = 1
            else:
                #me lost and me is 1
                actual_row["Y"] = 0
            
        else:
            for (key, value) in new_row_opp.items():
                actual_row[key + "_1"] = value
            
            for (key, value) in new_row_me.items():
                actual_row[key + "_2"] = value

            if info["winner"] == 1:
                #opp is 1 and opp lost 
                actual_row["Y"] = 0
            else:
                #opp is 1 and opp won
                actual_row["Y"] = 1
            
        new_rows.append(actual_row)
        
        i += 1

    df_complex = pd.DataFrame(new_rows)
        
    df_complex.to_csv(data_path)
    #standardize later

if __name__ == "__main__":
    datapath = "/home/scratch/24cjb4/complex.csv"
    if exists(datapath):
        print(datapath + " already exists!")
    else:
        convert(datapath)
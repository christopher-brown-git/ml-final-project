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

def convert(rows_of_data, data_path):
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
    out_vec = []

    for (key, info) in tqdm.tqdm(games_dict.items(), total=len(games_dict)):
        #until the end the first few columns is me, and the last few columns is opponent
        #decide order at end
        #outcome vector is 1 if the player in the first few columns beat the player in the last few columns

        new_row_1 = {}
        new_row_2 = {}

        cards_me = {}
        cards_opp = {}

        for (id, level) in info["me"]:
            card_name = card_dict[id][0] #FIX ERRORS LIKE THESE
            cards_me[card_name] = level
        
        for (id, level) in info["opp"]:
            card_name = card_dict[id][0]
            cards_opp[card_name] = level

        for r in [new_row_1, new_row_2]:
            r["elixir_cost"] = 0

            r["common_count"] = 0
            r['rare_count'] = 0
            r['epic_count'] = 0
            r['legendary_count'] = 0
            r['champion_count'] = 0
            r['spell_count'] = 0
            r['structure_count'] = 0
            r['spawner_count'] = 0
            r['melee_short_count'] = 0
            r['melee_medium_count'] = 0
            r['melee_long_count'] = 0
            r['ranged_count'] = 0
            r['air_count'] = 0
            r['ground_count'] = 0
            r['defensive_building_count'] = 0
            r['siege_building_count'] = 0
            r['total_card_level_count'] = 0
            r['troop_count'] = 0

            for f in other_features_map.values():
                r[f] = 0
            
            for f in other_features_map.values():
                r[f] = 0

        for (card_id, (card_name, elixir_cost)) in card_dict.items():
            #ignore mirror for now
            if card_name == "Mirror":
                if card_name in cards_me.keys():
                    new_row_1[card_name] = 1
                else:
                    new_row_1[card_name] = 0
                
                if card_name in cards_opp.keys():
                    new_row_2[card_name] = 1
                else:
                    new_row_2[card_name] = 1
                
                continue
            
            #ME/ROW 1

            if card_name in cards_me.keys():
                level_me = cards_me[card_name]
                new_row_1[card_name] = 1
                new_row_1[elixir_cost] += elixir_cost
                
                #handle counts
                for category in [common, rare, epic, legendary, champions, spells, 
                                 structures, spawners, melee_short, melee_medium, melee_long, ranged, air, ground,
                                 defensive_buildings, siege_buildings]:
                    
                    if card_name in category[1]:
                        new_row_1[category[0]] += 1
                    
                if card_name not in structures and card_name not in spells:
                    new_row_1["troop_count"] += 1
                
                new_row_1["total_card_level_count"] += level_me

                #go through other features:
                #if one of these features is a list, it should be accumulated
                #if it is a value, it should be averaged

                for feature in other_features_map.keys():
                    actual_feature = other_features_map[feature]
                    if feature in stats[card_name].keys():
                        mult = stats[card_name]['mult']
                        if isinstance(stats[card_name][feature], list):
                            index = level_me - stats[card_name]['Level'][0]
                            new_row_1[actual_feature] += stats[card_name][feature][index]*mult
                        else:
                            if new_row_1[actual_feature] == 0:
                                new_row_1[actual_feature] = []
                            new_row_1[actual_feature].append(stats[card_name][feature]*mult)

                #special case for spawners: just add the stats for all the troops the produce over their lifetime
                if card_name in spawners:
                    d = stats[card_name]
                    lifetime = d['Lifetime']
                    spawnspeed = d['SpawnSpeed']
                    spawnnumber = d['SpawnNumber']
                    spawnondeath = d['SpawnOnDeath']

                    num = int((lifetime//spawnspeed) * spawnnumber + spawnondeath)

                    for _ in range(0, num):
                        spawn = stats[card_name]['Spawn']
                        for feature in stats[spawn]:
                            #do not consider categorical features here
                            if feature in other_features_map.values():
                                if isinstance(stats[card_name][feature], list):
                                    #all spawners produce cards that are the same level as they are themselves
                                    index = level_me - stats[card_name]['Level'][0]
                                    mult = stats[card_name]['mult']
                                    new_row_1[feature] += stats[card_name][feature][index]*mult
                                else:
                                    if new_row_1[feature] == 0:
                                        new_row_1[feature] = []
                                    new_row_1[feature].append(stats[card_name][feature]*mult)
                
                #last special case
                if card_name == "Goblin Gang":
                        #go through features that accumulate (total)
                    for feature in stats["Goblins"].keys():
                        #do not consider categorical features here
                        if feature in other_features_map.values():
                            if isinstance(stats[card_name][feature], list):
                                #all spawners produce cards that are the same level as they are themselves
                                index = level_me - stats[card_name]['Level'][0]
                                mult = stats[card_name]['mult']
                                new_row_1[feature] += stats[card_name][feature][index]*mult
                            else:
                                if new_row_1[feature] == 0:
                                    new_row_1[feature] = []
                                new_row_1[feature].append(stats[card_name][feature]*mult)
                    
                    for feature in stats["Spear Goblins"].keys():
                        #do not consider categorical features here
                        if feature in other_features_map.values():
                            if isinstance(stats[card_name][feature], list):
                                #all spawners produce cards that are the same level as they are themselves
                                index = level_me - stats[card_name]['Level'][0]
                                mult = stats[card_name]['mult']
                                new_row_1[feature] += stats[card_name][feature][index]*mult
                            else:
                                if new_row_1[feature] == 0:
                                    new_row_1[feature] = []
                                new_row_1[feature].append(stats[card_name][feature]*mult)
            else:
                new_row_1[card_name] = 0
            
            
            #OPP/ROW 2
            if card_name in cards_opp.keys():
                level_opp = cards_opp[card_name]
                new_row_2[card_name] = 1
                new_row_2[elixir_cost] += elixir_cost
                
                #handle counts
                for category in [common, rare, epic, legendary, champions, spells, 
                                 structures, spawners, melee_short, melee_medium, melee_long, ranged, air, ground,
                                 defensive_buildings, siege_buildings]:
                    
                    if card_name in category[1]:
                        new_row_2[category[0]] += 1
                    
                if card_name not in structures and card_name not in spells:
                    new_row_2["troop_count"] += 1
                
                new_row_2["total_card_level_count"] += level_opp

                #go through other features:
                #if one of these features is a list, it should be accumulated
                #if it is a value, it should be averaged

                for feature in other_features_map.keys():
                    actual_feature = other_features_map[feature]
                    if feature in stats[card_name].keys():
                        mult = stats[card_name]['mult']
                        if isinstance(stats[card_name][feature], list):
                            index = level_opp - stats[card_name]['Level'][0]
                            new_row_2[actual_feature] += stats[card_name][feature][index]*mult
                        else:
                            if new_row_2[actual_feature] == 0:
                                new_row_2[actual_feature] = []
                            new_row_2[actual_feature].append(stats[card_name][feature]*mult)

                #special case for spawners: just add the stats for all the troops the produce over their lifetime
                if card_name in spawners:
                    d = stats[card_name]
                    lifetime = d['Lifetime']
                    spawnspeed = d['SpawnSpeed']
                    spawnnumber = d['SpawnNumber']
                    spawnondeath = d['SpawnOnDeath']

                    num = int((lifetime//spawnspeed) * spawnnumber + spawnondeath)

                    for _ in range(0, num):
                        spawn = stats[card_name]['Spawn']
                        for feature in stats[spawn]:
                            #do not consider categorical features here
                            if feature in other_features_map.values():
                                if isinstance(stats[card_name][feature], list):
                                    #all spawners produce cards that are the same level as they are themselves
                                    index = level_opp - stats[card_name]['Level'][0]
                                    mult = stats[card_name]['mult']
                                    new_row_2[feature] += stats[card_name][feature][index]*mult
                                else:
                                    if new_row_2[feature] == 0:
                                        new_row_2[feature] = []
                                    new_row_2[feature].append(stats[card_name][feature]*mult)
                
                #last special case
                if card_name == "Goblin Gang":
                        #go through features that accumulate (total)
                    for feature in stats["Goblins"].keys():
                        #do not consider categorical features here
                        if feature in other_features_map.values():
                            if isinstance(stats[card_name][feature], list):
                                #all spawners produce cards that are the same level as they are themselves
                                index = level_opp - stats[card_name]['Level'][0]
                                mult = stats[card_name]['mult']
                                new_row_2[feature] += stats[card_name][feature][index]*mult
                            else:
                                if new_row_2[feature] == 0:
                                    new_row_2[feature] = []
                                new_row_2[feature].append(stats[card_name][feature]*mult)
                    
                    for feature in stats["Spear Goblins"].keys():
                        #do not consider categorical features here
                        if feature in other_features_map.values():
                            if isinstance(stats[card_name][feature], list):
                                #all spawners produce cards that are the same level as they are themselves
                                index = level_opp - stats[card_name]['Level'][0]
                                mult = stats[card_name]['mult']
                                new_row_2[feature] += stats[card_name][feature][index]*mult
                            else:
                                if new_row_2[feature] == 0:
                                    new_row_2[feature] = []
                                new_row_2[feature].append(stats[card_name][feature]*mult)
            else:
                new_row_2[card_name] = 0
            

            
        
        me_comes_first = 0
        if np.random.rand() > 0.5:
            me_comes_first = 1

        #outcome is 1 if first player wins and 0 if they lose
        if me_comes_first:
            new_row_1.update(new_row_2)
            new_rows.append(new_row_1)
            if info["winner"] == 1:
                #I won and I come first
                out_vec.append(1)
            else:
                #I lost and I come first
                out_vec.append(0)
        else:
            new_row_2.update(new_row_1)
            new_rows.append(new_row_2)
            if info["winner"] == 1:
                #I won and I come second, so first player lost
                out_vec.append(0)
            else:
                #I lost and I come second, so first player won
                out_vec.append(1)
              
        #should be plenty of rows
        # if index >= rows_of_data:
        #     break

    df_small = pd.DataFrame(new_rows)
        
    df_small.to_csv(data_path)

if __name__ == "__main__":
    convert(100000, "comp.csv")
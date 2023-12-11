import requests
import pandas as pd
import tqdm
from bs4 import BeautifulSoup
import pickle
from os.path import exists

#USE DECK SHOP INSTEAD OF STATS ROYALE
URL = "https://www.deckshop.pro/card/list"
fp = "/home/scratch/24cjb4/webscraping/cardlist.html"

file_exists = exists(fp)

if not file_exists:
    page = requests.get(URL)
    with open(fp, 'wb+') as f:
        f.write(page.content)

soup = ""
with open(fp, 'rb') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

card_groups = soup.find_all("div", class_="mb-4 flex items-center")

names_to_link = {}
for card_group in card_groups:
    div_container = card_group.find("div", class_="flex flex-wrap items-center")
    
    a_containers = div_container.find_all("a", href=True)

    for a in a_containers:
        image = a.img
        name = "".join(image['alt'].split()) #no spaces in names
        link = a['href']
        card_url = URL[:-10]+link
        names_to_link[name] = card_url

stats = {}
special_names = set()
special_names.add("GoblinGang") #only contains link to goblins and spear goblins even tho this is technically innacurate

#EXTRA CATEGORIES
melee_short = {"Skeletons", "Goblins", "GoblinGang", "Barbarians", "Rascals", "ElixirGolem", "MiniP.E.K.K.A", 
                    "GoblinCage", "BarbarianBarrel", "GoblinBarrel", "SkeletonArmy", "GiantSkeleton", 
                    "Bandit", "Graveyard", "Lumberjack"}

melee_medium = {"Bats", "Knight", "EliteBarbarians", "Valkyrie", "DarkPrince", "GoblinGiant", 
                    "P.E.K.K.A", "ElectroGiant", "Miner", "RoyalGhost", "Fisherman", "MegaKnight", "GoldenKnight", 
                    "SkeletonKing", "Monk"}

melee_long = {"Minions", "RoyalDelivery", "MinionHorde", "RoyalRecruits", "MegaMinion", "BattleHealer", "Guards", 
                    "Prince", "Phoenix", "NightWitch", "MightyMiner"}

#ranged units have range tag already

air = {"Bats", "Minions", "SkeletonBarrel", "SkeletonDragons", "MinionHorde", "MegaMinion", "FlyingMachine", 
            "BabyDragon", "Balloon", "ElectroDragon", "InfernoDragon", "Phoenix", "LavaHound"}

ground = {"Skeletons", "ElectroSpirit", "FireSpirit", "IceSpirit", "Goblins", "SpearGoblins",
            "Bomber", "Archers", "Knight", "GoblinGang", "Firecracker", "RoyalDelivery", "Barbarians", "Rascals", "RoyalGiant",
            "EliteBarbarians", "RoyalRecruits", "HealSpirit", "IceGolem", "DartGoblin", "ElixirGolem", "MiniP.E.K.K.A", "Musketeer",
            "GoblinCage", "Valkyrie", "BattleRam", "HogRider", "BattleHealer", "Zappies", "Giant", "Wizard", "RoyalHogs",
            "ThreeMusketeers", "BarbarianBarrel", "WallBreakers", "GoblinBarrel", "Guards", "SkeletonArmy", "DarkPrince",
            "Hunter", "Witch", "Prince", "Bowler", "Executioner", "CannonCart", "GiantSkeleton", "GoblinGiant",
            "P.E.K.K.A", "ElectroGiant", "Golem", "Miner", "Princess", "IceWizard", "RoyalGhost", "Bandit", "Fisherman",
            "ElectroWizard", "MagicArcher", "Lumberjack", "NightWitch", "MotherWitch", "RamRider", "Sparky",
            "MegaKnight", "LittlePrince", "GoldenKnight", "SkeletonKing", "MightyMiner", "ArcherQueen", "Monk"}

defensive_buildings = {"Cannon", "Tesla", "BombTower", "InfernoTower"}

siege_buildings = {"Mortar", "X-Bow"}  

#speed 0 is for spells and buildings
speed_to_num = {}
speed_to_num["Slow"] = 1
speed_to_num["Medium"] = 2
speed_to_num["Fast"] = 3
speed_to_num["Very fast"] = 4

melee_range_to_num = {}

melee_range_to_num["NotMelee"] = 0
melee_range_to_num["Short"] = 1
melee_range_to_num["Medium"] = 2
melee_range_to_num["Long"] = 3

for name, link in tqdm.tqdm(names_to_link.items(), total=len(names_to_link)):
    URL = link
    fp = "/home/scratch/24cjb4/webscraping/" + name + ".html"
    file_exists = exists(fp)

    #save webpages bc I think the website is stopping me from sending too many requests for html files
    if not file_exists:
        page = requests.get(URL)
        with open(fp, 'wb+') as f:
            f.write(page.content)

    soup = ""
    with open(fp, 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    stats[name] = {}

    container_1 = soup.find("section", class_="mb-10")
    container_2 = container_1.find("div", class_="grid md:grid-cols-2 gap-5")
    
    if name in melee_short:
        stats[name]["MeleeRange"] = melee_range_to_num["Short"]
    elif name in melee_medium:
        stats[name]["MeleeRange"] = melee_range_to_num["Medium"]
    elif name in melee_long:
        stats[name]["MeleeRange"] = melee_range_to_num["Long"]
    else:
        stats[name]["MeleeRange"] = melee_range_to_num["NotMelee"]

    if name in air:
        stats[name]["Air"] = 1
    else:
        stats[name]["Air"] = 0

    if name in ground:
        stats[name]["Ground"] = 1
    else:
        stats[name]["Ground"] = 0

    if name in defensive_buildings:
        stats[name]["DefensiveBuilding"] = 1
    else:
        stats[name]["DefensiveBuilding"] = 0
    
    if name in siege_buildings:
        stats[name]["SiegeBuilding"] = 1
    else:
        stats[name]["SiegeBuilding"] = 0


    #special case for goblin gang, just hard code
    if name in special_names:
        goblins_soup = BeautifulSoup(requests.get("https://www.deckshop.pro/card/detail/goblins").content, "html.parser")
        spear_goblins_soup = BeautifulSoup(requests.get("https://www.deckshop.pro/card/detail/spear-goblins").content, "html.parser")
        stats[name]['Level'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['Goblins'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['SpearGoblins'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['mult'] = 3
        stats[name]['Speed'] = 4
    else:
        #normal case
        table = container_2.find("table")
        
        #collect categories of stats
        head = table.find("thead")

        col_headers = head.find_all("th")

        span_headers = []
        for c_h in col_headers:
            span_header = c_h.find("span", class_="hidden sm:inline")
            if span_header == None:
                other_card = c_h.find("a", href=True)
                if other_card == None:
                    span_headers.append("".join(c_h.text.split())) #Level
                else:
                    span_headers.append("".join(other_card.find("img")['alt'].split()))
            else:
                span_headers.append("".join(span_header.text.split()))

        stats_names = []
        for s_h in span_headers:
            if s_h is not None:
                #Knight -> {Level: [], Hitpoints: [], Damage: [], etc.}
                stats_names.append(s_h)
                stats[name][s_h] = []

        stats[name]["mult"] = 1 #for cards that consist of a group of the same cards, like skeletons or skeleton army or goblins

        #collect stats themselves
        body = table.find("tbody")
        rows = body.find_all("tr")

        for row in rows:
            cols_1 = [row.find("th")]
            cols_2 = row.find_all("td")
            cols = cols_1 + cols_2

            for i, col in enumerate(cols):
                if col.find("span") == None:
                    if 'Level' in col.text:
                        stats[name][stats_names[i]].append(int(col.text.split()[-1]))
                    else:
                        stats[name][stats_names[i]].append(int(col.text))
                else:
                    mirror_or_elite = col.find("span", class_="hidden sm:inline")
                    if mirror_or_elite and mirror_or_elite.text in {"(Mirrored)", "(Elite level)"}:
                        break

                    info = col.text.strip().split()
                    stat = int(info[0])
                    multiplier = int(info[1][1:])

                    stats[name][stats_names[i]].append(stat)
                    stats[name]["mult"] = multiplier
    
        section = soup.find("section", class_="bg-gradient-to-br from-gray-body to-gray-dark px-page py-3")
        outer_div = section.find("div", class_="grid md:grid-cols-2 gap-5")

        speed_and_other_stats = outer_div.find("table")
        
        for row in speed_and_other_stats.find_all("tr"):
            new_stat = "".join(row.find("th").text.split())
            if(new_stat not in stats[name].keys()):
                val = row.find("td").text.strip()
                if new_stat == "Speed":
                    stats[name][new_stat] = speed_to_num[val]
                else:
                    stats[name][new_stat] = float(val)

    
#special cases/incomplete data:
# spawn speed for Graveyard and Buildings that spawn cards (Tombstone, Furnace, Golbin Hut, Barbarian Hut, Goblin Drill)

stats["Graveyard"]["SpawnSpeed"] = 0.5

stats["Tombstone"]["SpawnSpeed"] = 4
stats["Tombstone"]["Lifetime"] = 30
stats["Tombstone"]["SpawnOnDeath"] = 4
stats["Tombstone"]["SpawnNumber"] = 2

stats["Furnace"]["SpawnSpeed"] = 5
stats["Furnace"]["Lifetime"] = 28
stats["Furnace"]["SpawnOnDeath"] = 1
stats["Furnace"]["SpawnNumber"] = 1

stats["GoblinHut"]["SpawnSpeed"] = 11
stats["GoblinHut"]["Lifetime"] = 29
stats["GoblinHut"]["SpawnOnDeath"] = 1
stats["GoblinHut"]["SpawnNumber"] = 3

stats["BarbarianHut"]["SpawnSpeed"] = 15
stats["BarbarianHut"]["Lifetime"] = 30
stats["BarbarianHut"]["SpawnOnDeath"] = 1
stats["BarbarianHut"]["SpawnNumber"] = 3

stats["GoblinDrill"]["SpawnSpeed"] = 3
stats["GoblinDrill"]["Lifetime"] = 9
stats["GoblinDrill"]["SpawnOnDeath"] = 2
stats["GoblinDrill"]["SpawnNumber"] = 1

f = open("stats.txt", "a")

unique_stats = set()
for name in stats.keys():
    f.write(name)
    f.write(str(stats[name]))
    unique_stats.update(stats[name].keys())
    f.write("\n")

f.close()


#write dictionary of statistics to a pkl file
with open('stats.pkl', 'wb') as file:
    pickle.dump(stats, file)        
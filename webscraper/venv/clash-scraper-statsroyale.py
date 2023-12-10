import requests
import pandas as pd
import tqdm
from bs4 import BeautifulSoup
import pickle

URL = "https://statsroyale.com/cards"
page = requests.get(URL)

#create BeautifulSoup object
soup = BeautifulSoup(page.content, "html.parser")

card_groups = soup.find_all("div", class_="cards__group")

names_to_link = {}
for card_group in card_groups:
    container = card_group.find("div", class_="cards__cards")
    
    cards = container.find_all("div", class_="cards__card")
    for card in cards:
        name = card.find("div", class_="ui__tooltip ui__tooltipTop ui__tooltipMiddle cards__tooltip").text
        link = card.find("a", href=True)
        names_to_link[name.strip()] = link['href'].strip()

#name -> (stats_category -> [level_1_stat, level_2_stat, etc.])

f = open("card-stats.txt", "a")
stats = {}
for name, link in tqdm.tqdm(names_to_link.items(), total=len(names_to_link)):
    URL = link
    page = requests.get(URL)

    stats[name] = {}
    
    #create beautiful soup object
    soup = BeautifulSoup(page.content, "html.parser")
    statistics = soup.find("div", class_="card__statistics")
    container = statistics.find("table", class_="card__desktopTable card__table")
    if container != None:
        rows = container.find_all("tr", class_="card__tableValues")

        if rows != None:
            for row in rows:

                # category = row.find("td", class_="card__tableValue").text.strip()
                if row != None:
                    cat_stats = row.find_all("td", class_="card__tableValue")
                    category = None

                    for c_s in cat_stats:
                        if c_s.find("img") != None:
                            category = c_s.text.strip()
                            stats[name][category] = []
                        else:
                            val = c_s.text.strip()
                            stats[name][category].append(val)

                    print(name, stats[name][category])
    # f.write(name)
    # f.write(stats[name])
    # f.write("**")

with open('stats.pkl', 'wb') as file:
    pickle.dump(stats, file)        
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
    
    #special case for goblin gang, just hard code
    if name in special_names:
        goblins_soup = BeautifulSoup(requests.get("https://www.deckshop.pro/card/detail/goblins").content, "html.parser")
        spear_goblins_soup = BeautifulSoup(requests.get("https://www.deckshop.pro/card/detail/spear-goblins").content, "html.parser")
        stats[name]['Level'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['Goblins'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['SpearGoblins'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        stats[name]['mult'] = 3
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
                    span_headers.append(c_h.text.strip()) #Level
                else:
                    span_headers.append("".join(other_card.find("img")['alt'].split()))
            else:
                span_headers.append(span_header.text)

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

print(stats['GoblinGang'])
#name -> (stats_category -> [level_1_stat, level_2_stat, etc.])

# f = open("card-stats.txt", "a")
# stats = {}
# for name, link in tqdm.tqdm(names_to_link.items(), total=len(names_to_link)):
#     URL = link
#     page = requests.get(URL)

#     stats[name] = {}
    
#     #create beautiful soup object
#     soup = BeautifulSoup(page.content, "html.parser")
#     statistics = soup.find("div", class_="card__statistics")
#     container = statistics.find("table", class_="card__desktopTable card__table")
#     if container != None:
#         rows = container.find_all("tr", class_="card__tableValues")

#         if rows != None:
#             for row in rows:

#                 # category = row.find("td", class_="card__tableValue").text.strip()
#                 if row != None:
#                     cat_stats = row.find_all("td", class_="card__tableValue")
#                     category = None

#                     for c_s in cat_stats:
#                         if c_s.find("img") != None:
#                             category = c_s.text.strip()
#                             stats[name][category] = []
#                         else:
#                             val = c_s.text.strip()
#                             stats[name][category].append(val)

#                     print(name, stats[name][category])
#     # f.write(name)
#     # f.write(stats[name])
#     # f.write("**")

# with open('stats.pkl', 'wb') as file:
#     pickle.dump(stats, file)        
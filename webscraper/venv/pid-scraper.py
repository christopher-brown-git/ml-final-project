import requests
import pandas as pd
import tqdm
from bs4 import BeautifulSoup
import pickle
from os.path import exists

#FIRST GET TOP 100 CLAN IDS FROM DECKSHOP.PRO
URL = "https://www.deckshop.pro/spy/top/global/clans"
fp = "/home/scratch/24cjb4/webscraping/topglobalclans.html"

file_exists = exists(fp)

if not file_exists:
    page = requests.get(URL)
    with open(fp, 'wb+') as f:
        f.write(page.content)

soup = ""
with open(fp, 'rb') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

m1 = soup.find("main", class_="page-container py-1 lg:py-2 xl:py-3")
a1 = m1.find("article", class_="px-page")
table = a1.find("table", class_="mb-5")

tbody = table.find("tbody")

trs = tbody.find_all("tr")

clan_ids = []
for tr in trs:
    clan_ids.append(tr['id'])

#GET NAMES OF MEMBERS IN CLANS USING CLASH ROYALE API

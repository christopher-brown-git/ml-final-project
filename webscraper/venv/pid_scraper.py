import requests
import tqdm
from bs4 import BeautifulSoup
import pickle
from os.path import exists
import json

def scrape():
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

    apikey = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjJjNDVhNWM1LTM2NDctNDFjZC05MTUwLTgzZDI4YmUxYzQzMyIsImlhdCI6MTcwMjQxMTY2Nywic3ViIjoiZGV2ZWxvcGVyL2NmNWJjOTAyLWFlNWEtMjM5OC0zNDkwLWVlZDkyOWEyYzNmYiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMzcuMTY1LjguMTAiLCIxMzcuMTY1LjEwNi4xMjYiXSwidHlwZSI6ImNsaWVudCJ9XX0.ufbPS9wibWsaEIvCy2e8tRTNH-yj1FKAxWiyHJfRipPz7GN0PzFw8IOrDpMP3viBTcZ4g2J4lxCisLn3ds0GOA"

    #me = VC2L0QPC
    pids = []

    for cid in tqdm.tqdm(clan_ids, total = len(clan_ids)):
        s_cid = str(cid)
        r=requests.get("https://api.clashroyale.com/v1/clans/%23" +s_cid + "/members", headers={"Accept":"application/json", "authorization":"Bearer " + apikey}, params = {"limit":50})
        res = r.json()
        for item in res['items']:
            pids.append(item['tag'])

    #write 5000 pids to pickle file

    #write dictionary of statistics to a pkl file
    with open('pids.pkl', 'wb') as file:
        pickle.dump(pids, file) 

if __name__ == "__main__":
    scrape()
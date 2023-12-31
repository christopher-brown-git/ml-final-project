import requests
import json
import tqdm
import pickle
from os.path import exists
import pid_scraper


def scrape_for_games(new_file_path):
    fp = "pids.pkl"

    file_exists = exists(fp)

    if not file_exists:
        pid_scraper.scrape()

    pids = ''
    with open('pids.pkl', 'rb') as file:
        pids = pickle.load(file)

    apikey = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjJjNDVhNWM1LTM2NDctNDFjZC05MTUwLTgzZDI4YmUxYzQzMyIsImlhdCI6MTcwMjQxMTY2Nywic3ViIjoiZGV2ZWxvcGVyL2NmNWJjOTAyLWFlNWEtMjM5OC0zNDkwLWVlZDkyOWEyYzNmYiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMzcuMTY1LjguMTAiLCIxMzcuMTY1LjEwNi4xMjYiXSwidHlwZSI6ImNsaWVudCJ9XX0.ufbPS9wibWsaEIvCy2e8tRTNH-yj1FKAxWiyHJfRipPz7GN0PzFw8IOrDpMP3viBTcZ4g2J4lxCisLn3ds0GOA"

    games = {}

    #25 appears to be the limit for number of games you can request
    i = 1
    k = 0
    for pid in tqdm.tqdm(pids, total=len(pids)):
        r=requests.get("https://api.clashroyale.com/v1/players/%23" + str(pid)[1:] + "/battlelog", headers={"Accept":"application/json", "authorization":"Bearer " + apikey}, params = {"limit":50})
        res = r.json()

        for game in res:
            battle_time = game["battleTime"]
            info_me = game["team"][0]
            info_opp = game["opponent"][0]

            cards_me = info_me["cards"]
            crowns_me = info_me["crowns"]

            cards_opp = info_opp["cards"]
            crowns_opp = info_opp["crowns"]

            #all cards start at level 1
            res_me = [(card['id'], card['level']) for card in cards_me]
            res_opp = [(card['id'], card['level']) for card in cards_opp]

            games[battle_time] = {"me": res_me,
                                    "opp": res_opp, 
                                    "winner": 1 if crowns_me == 3 else 0}
        
        i += 1

        if i % 500 == 0 or i - 1 == len(pids):
    
            with open(new_file_path.split(".")[0] + str(k) + ".pkl", 'wb') as file:
                pickle.dump(games, file)
                k += 1

if __name__ == "__main__":
    new_file = "/home/scratch/24cjb4/games1.pkl"

    if not exists(new_file):
        scrape_for_games(new_file)


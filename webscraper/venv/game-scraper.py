import requests
import json
from dotenv import dotenv_values

config = dotenv_values(".env")

apikey = config["APIKEY"]

#me = VC2L0QPC

r=requests.get("https://api.clashroyale.com/v1/players/%23QVC2PL8Q/battlelog", headers={"Accept":"application/json", "authorization":"Bearer " + apikey}, params = {"limit":100})
#res = json.dumps(r.json(), indent = 2)
res = r.json()

games = {}

print(len(res))
for game in res:
    print(game["team"][0]["cards"])

# for card_info in res[0]["team"][0]["cards"]:
#     print(card_info['name'], card_info['id'], card_info['level'], card_info['elixirCost'])
#     print("\n")
    
#print(res[0]["opponent"][0]["cards"])

#res[i]["opponent"][0]["crowns"] == 3 --> means this player won


import requests
import pickle
from os.path import exists
import json
from dotenv import dotenv_values

path = "card_dict.pkl"
def create_card_dict():

    if exists(path):
        print(path + " exists")
        return

    config = dotenv_values(".env")
    apikey = config["APIKEY"]

    #card_dict
    #card_code -> (card_name, elixir)

    card_dict = {}
    r=requests.get("https://api.clashroyale.com/v1/cards", headers={"Accept":"application/json", "authorization":"Bearer " + apikey}, params = {"limit":200})

    res = r.json()

    for item in res["items"]:
        card_dict[item["id"]] = (item["name"], item["elixirCost"])

    #write dictionary of statistics to a pkl file
    with open(path, 'wb') as file:
        pickle.dump(card_dict, file) 

if __name__ == "__main__":
    create_card_dict()
import requests
from bs4 import BeautifulSoup

URL = "https://statsroyale.com/cards"
page = requests.get(URL)

#create BeautifulSoup object
soup = BeautifulSoup(page.content, "html.parser")

card_groups = soup.find_all("div", class_="cards__group")

for card_groups in card_groups:
    print(card_groups, end="\n"*2)
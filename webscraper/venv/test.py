import pickle

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

stats = ''
with open('stats.pkl', 'rb') as file:
    stats = pickle.load(file)

l = [melee_short, melee_medium, melee_long, air, ground, defensive_buildings, siege_buildings]

for i in range(0, len(l)):
    for elem in l[i]:
        if elem not in stats.keys():
            print(elem, "error")
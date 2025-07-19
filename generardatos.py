import random
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).parent
INTERACTION_SCALE = 10          
SYNERGY_LIMIT = 500          
COUNTER_LIMIT = 500             

ver = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
champ_data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US/champion.json").json()["data"]
print(f"▶ Using LoL patch {ver}")


ROLE_MAP = {
    "Fighter":  "Top",
    "Tank":     "Top",
    "Mage":     "Mid",
    "Assassin": "Mid",
    "Marksman": "ADC",
    "Support":  "Support",
}

JUNGLE_LIST = [
    "Amumu", "Camille", "Diana", "Evelynn", "Fiddlesticks", "Gragas", "Graves", "Hecarim",
    "Ivern", "Jarvan IV", "Jax", "Kayn", "Kha'Zix", "Kindred", "Lee Sin", "Lillia", "Maokai",
    "Master Yi", "Nidalee", "Nocturne", "Nunu & Willump", "Olaf", "Rammus", "Rek'Sai", "Sejuani",
    "Shaco", "Shyvana", "Trundle", "Udyr", "Vi", "Viego", "Vladimir", "Volibear", "Warwick",
    "Xin Zhao", "Zac",
]


rows = []
for champ in champ_data.values():
    name = champ["name"]
    tag = champ["tags"][0]
    role = ROLE_MAP.get(tag, "Mid")  
    rows.append({"champion": name, "role": role})

df = pd.DataFrame(rows)

mask_jgl = df["champion"].isin(JUNGLE_LIST)
df.loc[mask_jgl, "role"] = "Jungle"
print("▶ Final role distribution:")
print(df["role"].value_counts())



def beta_trunc(alpha=50, beta=50, lo=0.45, hi=0.55):
    """Draw a Beta(alpha,beta) variate truncated to [lo,hi]."""
    while True:
        x = np.random.beta(alpha, beta)
        if lo <= x <= hi:
            return round(x, 4)

df["win_rate"] = [beta_trunc() for _ in df.index]

role_of = dict(zip(df["champion"], df["role"]))
champions = df["champion"].tolist()

random.shuffle(champions)  
pairs = list(itertools.combinations(champions, 2))
random.shuffle(pairs)


def is_synergy(a: str, b: str) -> bool:
    """Return True iff (a,b) qualifies as a cross‑role synergy pair."""
    ra, rb = role_of[a], role_of[b]
    return {ra, rb} in [{"Top", "Jungle"}, {"Mid", "Jungle"}, {"Support", "ADC"}]


def is_counter(a: str, b: str) -> bool:
    """Return True iff (a,b) qualifies as a counter (direction a → b)."""
    ra, rb = role_of[a], role_of[b]
    return {ra, rb} in [{"Mid", "Assassin"}, {"ADC", "Tank"}, {"Jungle", "Support"}]

synergy = {}
counter = {}
synergy_added = 0
counter_added = 0

for a, b in pairs:
    if synergy_added < SYNERGY_LIMIT and is_synergy(a, b):
        val = round(random.uniform(0.02, 0.05), 3)
        synergy[f"{a}|{b}"] = val
        synergy[f"{b}|{a}"] = val
        synergy_added += 1
        continue

    if counter_added < COUNTER_LIMIT and is_counter(a, b):
        val = -round(random.uniform(0.02, 0.05), 3)  
        counter[f"{a}|{b}"] = val  
        counter[f"{b}|{a}"] = 0    
        counter_added += 1
        continue

    if synergy_added >= SYNERGY_LIMIT and counter_added >= COUNTER_LIMIT:
        break

print(f"▶ Generated {synergy_added} synergy pairs (stored as {len(synergy)} directed entries)")
print(f"▶ Generated {counter_added} counter relations (stored as {len(counter)} directed entries)")

for d in (synergy, counter):
    for k in d.keys():
        d[k] = round(d[k] * INTERACTION_SCALE, 3)
print(f"▶ Interaction weights scaled by ×{INTERACTION_SCALE}")


csv_path = OUT / "datos_campeones.csv"
json_path = OUT / "interacciones.json"

df.to_csv(csv_path, index=False, encoding="utf-8")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"synergy": synergy, "counter": counter}, f, indent=2, ensure_ascii=False)

print("✅ Outputs written:")
print("   •", csv_path.relative_to(OUT))
print("   •", json_path.relative_to(OUT))

from pathlib import Path
import random
import itertools
import json
import requests
import numpy as np
import pandas as pd

# ============================================================
# PARÁMETROS
# ============================================================
SEED               = 42
INTERACTION_SCALE  = 10            # Escala multiplicativa final
SYNERGY_LIMIT      = 500           # Nº máximo de pares de sinergia (pares elegidos, no direcciones)
COUNTER_LIMIT      = 500           # Nº máximo de relaciones counter (pares elegidos)
COUNTER_SYMMETRIC  = False         # True => mismo valor negativo en ambas direcciones
WINRATE_BETA_ALPHA = 50
WINRATE_BETA_BETA  = 50
WINRATE_MIN        = 0.45
WINRATE_MAX        = 0.55

OUT = Path(__file__).parent

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# OBTENER LISTA DE CAMPEONES (Data Dragon)
# ============================================================
ver = requests.get(
    "https://ddragon.leagueoflegends.com/api/versions.json"
).json()[0]

champ_data = requests.get(
    f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US/champion.json"
).json()["data"]

print(f"▶ Using LoL patch {ver}")


ROLE_MAP = {
    "Fighter":  "Top",
    "Tank":     "Top",
    "Mage":     "Mid",
    "Assassin": "Mid",
    "Marksman": "ADC",
    "Support":  "Support",
}

# Lista manual de junglas
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
    tag = champ["tags"][0] if champ["tags"] else "Mage"
    role = ROLE_MAP.get(tag, "Mid")
    rows.append({"champion": name, "role": role})

df = pd.DataFrame(rows)

# Ajustar jungla
mask_jgl = df["champion"].isin(JUNGLE_LIST)
df.loc[mask_jgl, "role"] = "Jungle"

print("▶ Final role distribution:")
print(df["role"].value_counts())


def beta_trunc(alpha=WINRATE_BETA_ALPHA, beta=WINRATE_BETA_BETA,
               lo=WINRATE_MIN, hi=WINRATE_MAX):
    """Devuelve una muestra Beta(alpha,beta) truncada a [lo, hi]."""
    while True:
        x = np.random.beta(alpha, beta)
        if lo <= x <= hi:
            return round(float(x), 4)

df["win_rate"] = [beta_trunc() for _ in df.index]

role_of   = dict(zip(df["champion"], df["role"]))
champions = df["champion"].tolist()

# Para selección aleatoria reproducible
random.shuffle(champions)
pairs = list(itertools.combinations(champions, 2))
random.shuffle(pairs)


def is_synergy(a: str, b: str) -> bool:
    ra, rb = role_of[a], role_of[b]
    return {ra, rb} in [
        {"Top", "Jungle"},
        {"Mid", "Jungle"},
        {"Support", "ADC"},
    ]

def is_counter(a: str, b: str) -> bool:
    ra, rb = role_of[a], role_of[b]
    return {ra, rb} in [
        {"Mid", "Assassin"},
        {"ADC", "Tank"},
        {"Jungle", "Support"},
    ]


synergy = {}
counter = {}
synergy_added = 0
counter_added = 0

for a, b in pairs:
    # Primero intenta sinergia
    if synergy_added < SYNERGY_LIMIT and is_synergy(a, b):
        val = round(random.uniform(0.02, 0.05), 3)  # positivo
        # Sinergia simétrica: guardamos ambas direcciones
        synergy[f"{a}|{b}"] = val
        synergy[f"{b}|{a}"] = val
        synergy_added += 1
        continue  # pasa al siguiente par

    # Luego intenta counter
    if counter_added < COUNTER_LIMIT and is_counter(a, b):
        val = -round(random.uniform(0.02, 0.05), 3)  # NEGATIVO
        if COUNTER_SYMMETRIC:
            # Opción simétrica (misma desventaja en ambas direcciones)
            counter[f"{a}|{b}"] = val
            counter[f"{b}|{a}"] = val
        else:
            # Dirigido: una sola dirección a|b es counter; no añadimos b|a
            counter[f"{a}|{b}"] = val
        counter_added += 1
        continue

    # Si ya alcanzamos ambos límites, salimos
    if synergy_added >= SYNERGY_LIMIT and counter_added >= COUNTER_LIMIT:
        break

print(f"▶ Generated {synergy_added} synergy pairs (stored as {len(synergy)} directed entries)")
print(f"▶ Generated {counter_added} counter pairs "
      f"(stored as {len(counter)} directed entries; symmetric={COUNTER_SYMMETRIC})")


if INTERACTION_SCALE != 1:
    for d in (synergy, counter):
        for k in list(d.keys()):
            d[k] = round(d[k] * INTERACTION_SCALE, 3)
    print(f"▶ Interaction weights scaled by ×{INTERACTION_SCALE}")

# ============================================================
# GUARDAR ARCHIVOS
# ============================================================
csv_path = OUT / "datos_campeones.csv"
json_path = OUT / "interacciones.json"

df.to_csv(csv_path, index=False, encoding="utf-8")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"synergy": synergy, "counter": counter}, f,
              indent=2, ensure_ascii=False)

print("✅ Outputs written:")
print("   •", csv_path.name)
print("   •", json_path.name)


neg_counter_values = sum(1 for v in counter.values() if v < 0)
pos_counter_values = sum(1 for v in counter.values() if v > 0)
print(f"▶ Counter stats: negatives={neg_counter_values}, positives={pos_counter_values}")

neg_synergy_values = sum(1 for v in synergy.values() if v < 0)
print(f"▶ Synergy negatives (should be 0): {neg_synergy_values}")

print("Ejemplos synergy:", list(synergy.items())[:3])
print("Ejemplos counter:", list(counter.items())[:3])

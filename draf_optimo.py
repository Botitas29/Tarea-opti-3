from pathlib import Path
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pulp import (
    LpProblem, LpVariable, LpBinary, lpSum,
    LpMaximize, LpMinimize, PULP_CBC_CMD, value, LpStatus
)

# ------------------------------------------------------------
# 1. CONFIGURACIÓN -------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
CSV_PATH   = BASE_DIR / "datos_campeones.csv"
JSON_PATH  = BASE_DIR / "interacciones.json"

# ------------------------------------------------------------
# 2. CARGA DE DATOS ------------------------------------------
df = pd.read_csv(CSV_PATH)
with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)

def canon(a, b):
    return (a, b) if a < b else (b, a)
def safe_name(champ: str) -> str:
    return champ.replace(" ", "_").replace("'", "").replace(".", "")

# Unifica sinergias (+) y counters (–)
S = {}
for key, val in raw["synergy"].items():
    S[canon(*key.split("|"))] = val
for key, val in raw["counter"].items():
    S[canon(*key.split("|"))] = val

# Campeones por rol y win-rate
C_by_role  = df.groupby("role")["champion"].apply(list).to_dict()
w           = df.set_index("champion")["win_rate"].to_dict()
all_champs  = list(w.keys())

print("Roles detectados:", list(C_by_role.keys()))

# ------------------------------------------------------------
# 3. MILP COMPLETO (win-rate + sinergias/counters) ------------
def solve_milp(banned, locked):
    x = {c: LpVariable(f"x_{safe_name(c)}", cat=LpBinary) for c in w}
    y = {p: LpVariable(f"y_{safe_name(p[0])}_{safe_name(p[1])}", cat=LpBinary) for p in S}

    prob = LpProblem("Draft_Optimo", LpMaximize)
    prob += lpSum(w[c]*x[c] for c in x) + lpSum(S[p]*y[p] for p in y)

    # un campeón por rol
    for rol, champs in C_by_role.items():
        prob += lpSum(x[c] for c in champs) == 1

    # linealización interacciones
    for (a, b), var in y.items():
        prob += var <= x[a]
        prob += var <= x[b]
        prob += var >= x[a] + x[b] - 1

    # bans y locks
    for c in banned:
        prob += x[c] == 0
    for c, val in locked.items():
        prob += x[c] == val

    prob.solve(PULP_CBC_CMD(msg=False))
    return prob

model_milp = solve_milp([], {})
team_milp  = [c for c in all_champs if model_milp.variablesDict()[f"x_{safe_name(c)}"].value()==1]
score_milp = value(model_milp.objective)
print("Composición MILP:", team_milp)
print("Score MILP:", round(score_milp,4))
print("Estado MILP:", LpStatus[model_milp.status])

# ------------------------------------------------------------
# 4. PFCM PARA ELECCIÓN ÓPTIMA DE BANS -----------------------
def solve_bans_pfcm(team, S, B=5):
    candidates = [c for c in all_champs if c not in team]
    E, u, cost = [], {}, {}

    # s->c
    for c in candidates:
        E.append(("s",c)); u[("s",c)] = 1; cost[("s",c)] = 0
    # c->r
    for c in candidates:
        for r in team:
            E.append((c,r)); u[(c,r)] = 1
            pair = canon(r,c)
            cnt  = -S[pair] if S.get(pair,0)<0 else 0
            cost[(c,r)] = -cnt
    # r->t
    for r in team:
        E.append((r,"t")); u[(r,"t")] = B; cost[(r,"t")] = 0

    f = {
        (i,j): LpVariable(f"f_{safe_name(i)}_{safe_name(j)}", lowBound=0, upBound=u[(i,j)])
        for (i,j) in E
    }
    prob = LpProblem("Bans_PFCM", LpMinimize)
    prob += lpSum(cost[e]*f[e] for e in E)

    nodes = ["s"] + candidates + team + ["t"]
    for v in nodes:
        out = lpSum(f[(i,j)] for (i,j) in E if i==v)
        inn = lpSum(f[(i,j)] for (i,j) in E if j==v)
        if   v=="s": prob += out-inn == B
        elif v=="t": prob += out-inn == -B
        else:        prob += out-inn == 0

    prob.solve(PULP_CBC_CMD(msg=False))
    bans = [c for c in candidates if f[("s",c)].value()>0.5]
    strength = -value(prob.objective)
    return bans, strength, E, f

bans_pfcm, strength_pfcm, E_pfcm, f_pfcm = solve_bans_pfcm(team_milp, S, B=5)
print("Bans recomendados (PFCM):", bans_pfcm)
print("Fuerza total de bans:", round(strength_pfcm,4))

# ------------------------------------------------------------
# 5. GRAFICA PFCM – CON ETIQUETAS Y LEYENDA -------------------
G_ban = nx.DiGraph()
G_ban.add_node("s"); G_ban.add_node("t")
cands = [c for c in all_champs if c not in team_milp]
G_ban.add_nodes_from(cands); G_ban.add_nodes_from(team_milp)
for (i,j) in E_pfcm:
    G_ban.add_edge(i,j)

pos = nx.spring_layout(G_ban, k=1.0, iterations=200)
fig, ax = plt.subplots(figsize=(14,10))
ax.set_title("PFCM – Bans Óptimos", fontsize=16)

# aristas posibles
nx.draw_networkx_edges(G_ban, pos,
    edge_color="gray", alpha=0.1, arrows=True, label="Aristas posibles", ax=ax)
# aristas solución
sol = [(i,j) for (i,j) in E_pfcm if f_pfcm[(i,j)].value()>0.5]
nx.draw_networkx_edges(G_ban, pos, edgelist=sol,
    edge_color="red", width=2, arrows=True, label="Aristas solución", ax=ax)

# nodos con etiqueta
nx.draw_networkx_nodes(G_ban, pos, nodelist=["s"],
    node_color="lightgreen", node_size=80, label="Fuente (s)", ax=ax)
nx.draw_networkx_nodes(G_ban, pos, nodelist=["t"],
    node_color="lightcoral", node_size=80, label="Sumidero (t)", ax=ax)
nx.draw_networkx_nodes(G_ban, pos, nodelist=cands,
    node_color="gray", node_size=50, alpha=0.3, label="Candidatos", ax=ax)
nx.draw_networkx_nodes(G_ban, pos, nodelist=bans_pfcm,
    node_color="red", node_size=80, edgecolors="black", label="Bans seleccionados", ax=ax)
nx.draw_networkx_nodes(G_ban, pos, nodelist=team_milp,
    node_shape="s", node_color="gold", node_size=100, edgecolors="black", label="Picks MILP", ax=ax)

# etiquetas de nodos
labels = {**{c:c for c in cands}, **{r:r for r in team_milp}, "s":"s","t":"t"}
nx.draw_networkx_labels(G_ban, pos, labels, font_size=8, ax=ax)

# leyenda
ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
ax.axis("off"); plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 6. GRAFICA MILP – SPRING LAYOUT con etiquetas y leyenda -----
G_milp = nx.Graph()
G_milp.add_nodes_from(all_champs)
for rol,champs in C_by_role.items():
    G_milp.add_node(rol)
    for c in champs:
        G_milp.add_edge(rol,c, type="assign")
for (u,v), wgt in S.items():
    G_milp.add_edge(u,v, type="syn" if wgt>0 else "cnt", weight=abs(wgt))

assign = [(u,v) for u,v,d in G_milp.edges(data=True) if d['type']=="assign"]
syn    = [(u,v) for u,v,d in G_milp.edges(data=True) if d['type']=="syn"]
cnt    = [(u,v) for u,v,d in G_milp.edges(data=True) if d['type']=="cnt"]

pos_s = nx.spring_layout(G_milp, k=0.25, iterations=300)
fig, ax = plt.subplots(figsize=(14,12))
ax.set_title("Spring Layout – MILP", fontsize=16)

# aristas
nx.draw_networkx_edges(G_milp, pos_s, edgelist=assign,
    edge_color="gray", alpha=0.3, width=0.8, label="Asignación", ax=ax)
nx.draw_networkx_edges(G_milp, pos_s, edgelist=syn,
    style="dashed", edge_color="forestgreen", alpha=0.6, width=0.8, label="Sinergia", ax=ax)
nx.draw_networkx_edges(G_milp, pos_s, edgelist=cnt,
    style="dotted", edge_color="firebrick",   alpha=0.6, width=0.8, label="Counter", ax=ax)

# nodos
nx.draw_networkx_nodes(G_milp, pos_s, nodelist=all_champs,
    node_color="lightblue", alpha=0.8,
    node_size=[200+1000*(w[c]-0.45) for c in all_champs],
    label="Campeones", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_s, nodelist=list(C_by_role.keys()),
    node_shape="s", node_color="slateblue", alpha=0.9, node_size=200,
    label="Roles", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_s, nodelist=team_milp,
    node_color="gold", edgecolors="black", node_size=200,
    label="Picks MILP", ax=ax)

nx.draw_networkx_labels(G_milp, pos_s,
    labels={c:c for c in team_milp}, font_weight="bold", ax=ax)

# leyenda
ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
ax.axis("off"); plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 7. GRAFICA MILP – SHELL LAYOUT con líneas continuas y leyenda
pos_h = nx.shell_layout(G_milp,
    nlist=[list(C_by_role.keys()), all_champs])
fig, ax = plt.subplots(figsize=(14,12))
ax.set_title("Shell Layout – MILP (líneas continuas)", fontsize=2)

# todas las aristas en línea continua
nx.draw_networkx_edges(
    G_milp, pos_h, edgelist=assign,
    edge_color="gray", alpha=0.3, width=0.8,
    style="solid", label="Asignación", ax=ax
)
nx.draw_networkx_edges(
    G_milp, pos_h, edgelist=syn,
    edge_color="forestgreen", alpha=0.6, width=0.1,
    style="solid", label="Sinergia", ax=ax
)
nx.draw_networkx_edges(
    G_milp, pos_h, edgelist=cnt,
    edge_color="firebrick", alpha=0.6, width=0.1,
    style="solid", label="Counter", ax=ax
)

# nodos
nx.draw_networkx_nodes(
    G_milp, pos_h, nodelist=all_champs,
    node_color="lightblue", alpha=0.8,
    node_size=[200+1000*(w[c]-0.45) for c in all_champs],
    label="Campeones", ax=ax
)
nx.draw_networkx_nodes(
    G_milp, pos_h, nodelist=list(C_by_role.keys()),
    node_shape="s", node_color="slateblue", alpha=0.9,
    node_size=200, label="Roles", ax=ax
)
nx.draw_networkx_nodes(
    G_milp, pos_h, nodelist=team_milp,
    node_color="gold", edgecolors="black",
    node_size=200, label="Picks MILP", ax=ax
)

nx.draw_networkx_labels(
    G_milp, pos_h,
    labels={c:c for c in team_milp},
    font_weight="bold", ax=ax
)

# leyenda
ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
ax.axis("off")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. RESULTADOS EN IMAGEN – Comparación de scores
fig = plt.figure(figsize=(12,6))

# a) panel de texto
ax_txt = fig.add_subplot(1,2,1)
ax_txt.axis("off")
info = (
    f"Picks MILP: {team_milp}\n"
    f"Score MILP: {score_milp:.4f}\n\n"
    f"Bans PFCM: {bans_pfcm}\n"
    f"Fuerza bans: {strength_pfcm:.4f}"
)
ax_txt.text(0, 0.5, info, fontsize=12, family="monospace", va="center")

# b) gráfico de barras
ax_bar = fig.add_subplot(1,2,2)
models = ["MILP", "PFCM"]
scores = [score_milp, strength_pfcm]
bars = ax_bar.bar(models, scores, color=["gold","red"])
ax_bar.set_title("Comparación de valores objetivo", fontsize=14)
ax_bar.set_ylabel("Valor objetivo")
for bar, sc in zip(bars, scores):
    ax_bar.text(
        bar.get_x() + bar.get_width()/2, sc + 0.01,
        f"{sc:.4f}", ha="center", va="bottom", fontsize=12
    )

plt.tight_layout()
plt.show()

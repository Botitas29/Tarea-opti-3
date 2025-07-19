"""
draft_optimo.py
Optimiza picks (MILP) y bans (PFCM) para un draft de League of Legends.
Incluye versión debug del modelo de bans con fallback greedy.
Requisitos: pandas, matplotlib, networkx, pulp

Archivos esperados en la misma carpeta:
    - datos_campeones.csv   (columnas: champion, role, win_rate)
    - interacciones.json    (claves: "synergy", "counter"; cada una dict "A|B": valor)
"""

# ------------------------------------------------------------
# 0. IMPORTS --------------------------------------------------
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pulp import (
    LpProblem, LpVariable, LpBinary, lpSum,
    LpMaximize, LpMinimize, PULP_CBC_CMD, value, LpStatus
)

# ------------------------------------------------------------
# 1. CONFIGURACIÓN -------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
CSV_PATH   = BASE_DIR / "datos_campeones.csv"
JSON_PATH  = BASE_DIR / "interacciones.json"

# Flags de depuración
DEBUG_BANS_VERBOSE_SOLVER = False   # True para ver salida completa de CBC
DEBUG_PRINT_FIRST_NEG     = 10      # cuántos pares negativos mostrar (0 = ninguno)
FALLBACK_GREEDY           = True    # usar greedy si modelo de bans no es óptimo

# ------------------------------------------------------------
# 2. CARGA DE DATOS ------------------------------------------
df = pd.read_csv(CSV_PATH)
with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)

def canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def safe_name(champ: str) -> str:
    return (champ.replace(" ", "_")
                 .replace("'", "")
                 .replace(".", "")
                 .replace("&", "and"))

# Unifica sinergias (+) y counters (−) en un solo diccionario S
S: dict[tuple[str, str], float] = {}
for key, val in raw["synergy"].items():
    S[canon(*key.split("|"))] = val
for key, val in raw["counter"].items():
    S[canon(*key.split("|"))] = val

# Campeones por rol y win-rate
C_by_role  = df.groupby("role")["champion"].apply(list).to_dict()
w          = df.set_index("champion")["win_rate"].to_dict()
all_champs = list(w.keys())

print("Roles detectados:", list(C_by_role.keys()))

# Estadísticas rápidas sobre interacciones
neg_total = sum(1 for v in S.values() if v < 0)
pos_total = sum(1 for v in S.values() if v > 0)
print(f"Total pares sinergia/counter: {len(S)}  (positivos={pos_total}, negativos={neg_total})")

if DEBUG_PRINT_FIRST_NEG > 0:
    ejemplos_neg = [(a, b, v) for (a, b), v in S.items() if v < 0][:DEBUG_PRINT_FIRST_NEG]
    print(f"Ejemplos negativos (hasta {DEBUG_PRINT_FIRST_NEG}): {ejemplos_neg}")

# ------------------------------------------------------------
# 3. MILP COMPLETO (win‑rate + sinergias/counters) ------------
def solve_milp(banned: list[str], locked: dict[str, int | float]):
    x = {c: LpVariable(f"x_{safe_name(c)}", cat=LpBinary) for c in w}
    y = {p: LpVariable(f"y_{safe_name(p[0])}_{safe_name(p[1])}", cat=LpBinary) for p in S}

    prob = LpProblem("Draft_Optimo", LpMaximize)
    prob += lpSum(w[c] * x[c] for c in x) + lpSum(S[p] * y[p] for p in y)

    # Un campeón por rol
    for rol, champs in C_by_role.items():
        prob += lpSum(x[c] for c in champs) == 1

    # Linealización interacciones
    for (a, b), var in y.items():
        prob += var <= x[a]
        prob += var <= x[b]
        prob += var >= x[a] + x[b] - 1

    # Bans y locks
    for c in banned:
        if c in x:
            prob += x[c] == 0
    for c, val in locked.items():
        if c in x:
            prob += x[c] == val

    prob.solve(PULP_CBC_CMD(msg=False))
    return prob

model_milp = solve_milp([], {})
team_milp  = [c for c in all_champs if model_milp.variablesDict()[f"x_{safe_name(c)}"].value() == 1]
score_milp = value(model_milp.objective)
print("Composición MILP:", team_milp)
print("Score MILP:", round(score_milp, 4))
print("Estado MILP:", LpStatus[model_milp.status])

# ------------------------------------------------------------
# 4. PFCM PARA ELECCIÓN ÓPTIMA DE BANS (versión debug) -------
def solve_bans_pfcm(team: list[str],
                    S: dict[tuple[str, str], float],
                    B: int = 5,
                    verbose: bool = False,
                    greedy_fallback: bool = True):
    """
    Devuelve (bans, strength, E, f) usando un PFCM.
    strength = suma de fuerzas de counters eliminados (positiva).
    Si el modelo no llega a óptimo y greedy_fallback=True se recurre a un método greedy.
    """
    team_set   = set(team)
    candidates = [c for c in all_champs if c not in team_set]

    # Chequeo cardinalidad
    if len(candidates) < B:
        print(f"[ERROR] len(candidates)={len(candidates)} < B={B}. Ajustando B.")
        B = len(candidates)

    # Contar counters negativos relevantes (candidato vs pick)
    neg_pairs = [
        (a, b, v) for (a, b), v in S.items() if v < 0 and (
            (a in team_set and b in candidates) or (b in team_set and a in candidates)
        )
    ]
    print(f"[DEBUG] counters relevantes (negativos) = {len(neg_pairs)}")

    # Si no hay counters, la fuerza es 0 (no hay nada que optimizar)
    if len(neg_pairs) == 0:
        print("[ADVERTENCIA] No hay interacciones negativas entre picks y candidatos.")
        chosen = candidates[:B]
        return chosen, 0.0, [], {}

    # Construcción del grafo de flujo
    E, u, cost = [], {}, {}
    # s -> c
    for c in candidates:
        E.append(("s", c)); u[("s", c)] = 1; cost[("s", c)] = 0
    # c -> r
    for c in candidates:
        for r in team:
            E.append((c, r)); u[(c, r)] = 1
            pair = canon(r, c)
            cnt  = -S[pair] if S.get(pair, 0) < 0 else 0  # cnt >= 0
            cost[(c, r)] = -cnt  # costo ≤ 0
    # r -> t
    for r in team:
        E.append((r, "t")); u[(r, "t")] = B; cost[(r, "t")] = 0

    f = {
        (i, j): LpVariable(f"f_{safe_name(i)}_{safe_name(j)}", lowBound=0, upBound=u[(i, j)])
        for (i, j) in E
    }

    prob = LpProblem("Bans_PFCM", LpMinimize)
    prob += lpSum(cost[e] * f[e] for e in E)

    nodes = ["s"] + candidates + team + ["t"]
    for v in nodes:
        out = lpSum(f[(i, j)] for (i, j) in E if i == v)
        inn = lpSum(f[(i, j)] for (i, j) in E if j == v)
        if v == "s":
            prob += out - inn == B
        elif v == "t":
            prob += out - inn == -B
        else:
            prob += out - inn == 0

    prob.solve(PULP_CBC_CMD(msg=verbose))
    status_str = LpStatus[prob.status]
    obj_val = value(prob.objective)
    print(f"[DEBUG] Estado modelo bans: {status_str}")
    print(f"[DEBUG] Valor objetivo raw (minimización, debería ≤0): {obj_val}")

    if status_str != "Optimal" or obj_val is None:
        print("[ADVERTENCIA] Modelo de bans no óptimo "
              f"(estado={status_str}, obj={obj_val}).")
        if greedy_fallback:
            # Greedy: escoge B candidatos con mayor suma de counters negativos contra el team
            contrib = {}
            for c in candidates:
                s_c = 0.0
                for r in team:
                    pair = canon(c, r)
                    v = S.get(pair, 0)
                    if v < 0:  # v negativo => counter
                        s_c += -v
                contrib[c] = s_c
            ordered = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
            bans = [c for c, sc in ordered[:B]]
            strength = sum(contrib[c] for c in bans)
            print(f"[FALLBACK] Bans greedy = {bans} (fuerza={strength:.4f})")
            return bans, strength, E, f
        else:
            raise ValueError("Modelo de bans no óptimo y fallback desactivado.")

    # Reconstruir solución óptima
    bans = [c for c in candidates if f[("s", c)].value() > 0.5]
    strength = -obj_val  # convertir costo (≤0) a fuerza positiva
    return bans, strength, E, f

# Llamada principal (ajusta B según candidatos)
B_BANS = 5
B_BANS = min(B_BANS, len([c for c in all_champs if c not in team_milp]))

bans_pfcm, strength_pfcm, E_pfcm, f_pfcm = solve_bans_pfcm(
    team_milp, S, B=B_BANS,
    verbose=DEBUG_BANS_VERBOSE_SOLVER,
    greedy_fallback=FALLBACK_GREEDY
)
print("Bans recomendados (PFCM):", bans_pfcm)
print("Fuerza total de bans:", round(strength_pfcm, 4))

# ------------------------------------------------------------
# 5. GRÁFICA PFCM --------------------------------------------
if E_pfcm:  # si hubo modelo (no solo fallback trivial sin aristas)
    G_ban = nx.DiGraph()
    G_ban.add_nodes_from(["s", "t"])
    cands = [c for c in all_champs if c not in team_milp]
    G_ban.add_nodes_from(cands)
    G_ban.add_nodes_from(team_milp)
    for (i, j) in E_pfcm:
        G_ban.add_edge(i, j)

    pos_ban = nx.spring_layout(G_ban, k=1.0, iterations=200, seed=42)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title("PFCM – Bans óptimos", fontsize=16)

    nx.draw_networkx_edges(G_ban, pos_ban,
        edge_color="gray", alpha=0.1, arrows=True, label="Aristas posibles", ax=ax)
    if f_pfcm:
        sol_edges = [(i, j) for (i, j) in E_pfcm if f_pfcm[(i, j)].value() > 0.5]
        nx.draw_networkx_edges(G_ban, pos_ban, edgelist=sol_edges,
            edge_color="red", width=2, arrows=True, label="Aristas solución", ax=ax)

    nx.draw_networkx_nodes(G_ban, pos_ban, nodelist=["s"],
        node_color="lightgreen", node_size=80, label="Fuente (s)", ax=ax)
    nx.draw_networkx_nodes(G_ban, pos_ban, nodelist=["t"],
        node_color="lightcoral", node_size=80, label="Sumidero (t)", ax=ax)
    nx.draw_networkx_nodes(G_ban, pos_ban, nodelist=cands,
        node_color="gray", node_size=50, alpha=0.3, label="Candidatos", ax=ax)
    nx.draw_networkx_nodes(G_ban, pos_ban, nodelist=bans_pfcm,
        node_color="red", node_size=80, edgecolors="black", label="Bans seleccionados", ax=ax)
    nx.draw_networkx_nodes(G_ban, pos_ban, nodelist=team_milp,
        node_shape="s", node_color="gold", node_size=100, edgecolors="black", label="Picks MILP", ax=ax)

    labels_ban = {c: c for c in cands} | {r: r for r in team_milp} | {"s": "s", "t": "t"}
    nx.draw_networkx_labels(G_ban, pos_ban, labels_ban, font_size=8, ax=ax)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("[INFO] No se genera gráfica de flujo (fallback trivial sin aristas).")

# ------------------------------------------------------------
# 6. GRÁFICA MILP – spring layout ----------------------------
G_milp = nx.Graph()
G_milp.add_nodes_from(all_champs)
for rol, champs in C_by_role.items():
    G_milp.add_node(rol)
    for c in champs:
        G_milp.add_edge(rol, c, type="assign")
for (u, v), wgt in S.items():
    G_milp.add_edge(u, v, type="syn" if wgt > 0 else "cnt", weight=abs(wgt))

assign_edges = [(u, v) for u, v, d in G_milp.edges(data=True) if d["type"] == "assign"]
syn_edges    = [(u, v) for u, v, d in G_milp.edges(data=True) if d["type"] == "syn"]
cnt_edges    = [(u, v) for u, v, d in G_milp.edges(data=True) if d["type"] == "cnt"]

pos_spring = nx.spring_layout(G_milp, k=0.25, iterations=300, seed=42)
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_title("Spring layout – MILP", fontsize=16)

nx.draw_networkx_edges(G_milp, pos_spring, edgelist=assign_edges,
    edge_color="gray", alpha=0.3, width=0.8, label="Asignación", ax=ax)
nx.draw_networkx_edges(G_milp, pos_spring, edgelist=syn_edges,
    style="dashed", edge_color="forestgreen", alpha=0.6, width=0.8, label="Sinergia", ax=ax)
nx.draw_networkx_edges(G_milp, pos_spring, edgelist=cnt_edges,
    style="dotted", edge_color="firebrick", alpha=0.6, width=0.8, label="Counter", ax=ax)

nx.draw_networkx_nodes(G_milp, pos_spring, nodelist=all_champs,
    node_color="lightblue", alpha=0.8,
    node_size=[200 + 1000 * (w[c] - 0.45) for c in all_champs],
    label="Campeones", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_spring, nodelist=list(C_by_role.keys()),
    node_shape="s", node_color="slateblue", alpha=0.9, node_size=200,
    label="Roles", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_spring, nodelist=team_milp,
    node_color="gold", edgecolors="black", node_size=200,
    label="Picks MILP", ax=ax)

nx.draw_networkx_labels(G_milp, pos_spring,
    labels={c: c for c in team_milp}, font_weight="bold", ax=ax)

ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
ax.axis("off")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. GRÁFICA MILP – shell layout -----------------------------
pos_shell = nx.shell_layout(G_milp, nlist=[list(C_by_role.keys()), all_champs])
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_title("Shell layout – MILP (líneas continuas)", fontsize=16)

nx.draw_networkx_edges(G_milp, pos_shell, edgelist=assign_edges,
    edge_color="gray", alpha=0.3, width=0.8, style="solid", label="Asignación", ax=ax)
nx.draw_networkx_edges(G_milp, pos_shell, edgelist=syn_edges,
    edge_color="forestgreen", alpha=0.6, width=0.1, style="solid", label="Sinergia", ax=ax)
nx.draw_networkx_edges(G_milp, pos_shell, edgelist=cnt_edges,
    edge_color="firebrick", alpha=0.6, width=0.1, style="solid", label="Counter", ax=ax)

nx.draw_networkx_nodes(G_milp, pos_shell, nodelist=all_champs,
    node_color="lightblue", alpha=0.8,
    node_size=[200 + 1000 * (w[c] - 0.45) for c in all_champs],
    label="Campeones", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_shell, nodelist=list(C_by_role.keys()),
    node_shape="s", node_color="slateblue", alpha=0.9, node_size=200,
    label="Roles", ax=ax)
nx.draw_networkx_nodes(G_milp, pos_shell, nodelist=team_milp,
    node_color="gold", edgecolors="black", node_size=200,
    label="Picks MILP", ax=ax)

nx.draw_networkx_labels(G_milp, pos_shell,
    labels={c: c for c in team_milp},
    font_weight="bold", ax=ax)

ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
ax.axis("off")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. RESULTADOS EN IMAGEN – comparación de scores ------------
fig = plt.figure(figsize=(12, 6))

ax_txt = fig.add_subplot(1, 2, 1)
ax_txt.axis("off")
info = (
    f"Picks MILP: {team_milp}\n"
    f"Score MILP: {score_milp:.4f}\n\n"
    f"Bans PFCM/Greedy: {bans_pfcm}\n"
    f"Fuerza bans: {strength_pfcm:.4f}"
)
ax_txt.text(0, 0.5, info, fontsize=12, family="monospace", va="center")

ax_bar = fig.add_subplot(1, 2, 2)
models = ["MILP", "Bans"]
scores = [score_milp, strength_pfcm]
bars = ax_bar.bar(models, scores, color=["gold", "red"])
ax_bar.set_title("Comparación de valores", fontsize=14)
ax_bar.set_ylabel("Valor")
for bar, sc in zip(bars, scores):
    ax_bar.text(bar.get_x() + bar.get_width()/2, sc + 0.01,
                f"{sc:.4f}", ha="center", va="bottom", fontsize=12)

plt.tight_layout()
plt.show()

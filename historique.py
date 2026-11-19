# IATMO : calage sur l'historique et performances quotidiennes, à partir de l'archive de la branche « donnees ».
# Lancé une fois par jour par la tâche GitHub (.github/workflows/copie_lcsqa.yml), après la copie.
#   python3 historique.py <dossier de la branche donnees> <dossier contenant stations_ref.json et fiches_stations.json>
# Écrit :
#   calage/calage.json      : incrément trafic − fond, profil horaire (jours ouvrés, week-end), exposant du vent, effet de la pluie,
#                             exposant du trafic et confinement calés sur les stations documentées, sur les 30 derniers jours complets ;
#   suivi/performances.json : pour chaque jour complet, validation croisée du krigeage ordinaire (station retirée tour à tour) par
#                             polluant, taux de bonne classe, MQI de FAIRMODE sur 7 et 30 jours glissants, et bilan des copies (90 jours) ;
#   suivi/mqi_jours.json    : sommes quotidiennes par station et par polluant pour le MQI (30 jours) ;
#   suivi/mqi_fenetres.json : MQI par station et MQI du 90e centile sur les 7 et 30 derniers jours (diagramme cible de la page).
# MQI : guide FAIRMODE version 4.0 (mai 2026, tableau 1, directive 2024/2881) ; NO₂ horaire, O₃ maximum journalier sur 8 h,
#       PM₁₀ et PM₂,₅ moyenne journalière. Paramètres à revoir avec la mise à jour annoncée pour fin 2026.
# Heures : celles des fichiers du LCSQA (temps universel en métropole, heure locale dans les DROM) sont converties en heure de Paris.
import csv, gzip, glob, json, math, os, sys, time, unicodedata, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np

T0 = time.time()
OUT, REF = sys.argv[1], sys.argv[2]
PARIS = ZoneInfo("Europe/Paris"); UTC = dt.timezone.utc
AUJ = dt.date.fromisoformat(os.environ["IATMO_JOUR"]) if os.environ.get("IATMO_JOUR") else dt.datetime.now(PARIS).date()   # IATMO_JOUR : essais
POLS = ["NO2", "O3", "PM10", "PM25"]
GRILLE_AEE = {"NO2": [10, 25, 60, 100, 150], "O3": [60, 100, 120, 160, 180], "PM10": [15, 45, 120, 195, 270], "PM25": [5, 15, 50, 90, 140]}
# FAIRMODE 4.0, tableau 1 : valeur limite LV (µg/m³), U_r,fix, β court terme, plancher U∅ (µg/m³), période d'évaluation
FM4 = {"NO2": (200, 0.15, 3.2, 3, "h"), "O3": (120, 0.15, 2.2, 10, "8h"), "PM10": (45, 0.25, 2.2, 4, "j"), "PM25": (25, 0.25, 2.5, 3, "j")}
def u_obs(p, O):
    """Incertitude de mesure : U∅ à concentration nulle, croissance linéaire jusqu'à U_r·LV à la valeur limite, puis proportionnelle."""
    lv, ur, _, u0, _ = FM4[p]; O = np.asarray(O, float)
    return np.where(O <= lv, u0 + (ur * lv - u0) * O / lv, ur * O)
def sommes(p, serie):
    """Paires (mesure, estimation) de la période d'évaluation d'un jour → sommes n, ΣO, ΣM, ΣO², ΣM², ΣOM, ΣU² (ou None)."""
    per = FM4[p][4]
    if per == "h": paires = [(o, m) for _, o, m in serie]
    elif per == "j":
        if len(serie) < 18: return None                                     # 75 % des heures
        paires = [(float(np.mean([o for _, o, _ in serie])), float(np.mean([m for _, _, m in serie])))]
    else:                                                                   # O₃ : maximum des moyennes glissantes sur 8 h (6 heures au moins)
        par_h = {h: (o, m) for h, o, m in serie}; mo, mm = [], []
        for d in range(0, 17):
            v = [par_h[h] for h in range(d, d + 8) if h in par_h]
            if len(v) >= 6: mo.append(np.mean([a for a, _ in v])); mm.append(np.mean([b for _, b in v]))
        if len(mo) < 13: return None                                        # 75 % des 17 moyennes du jour
        paires = [(float(max(mo)), float(max(mm)))]
    if not paires: return None
    O = np.array([a for a, _ in paires]); M = np.array([b for _, b in paires]); U = u_obs(p, O)
    return [len(paires), float(O.sum()), float(M.sum()), float((O * O).sum()), float((M * M).sum()), float((O * M).sum()), float((U * U).sum())]
def mqi_station(p, S):
    n, so, sm, soo, smm, som, su2 = S
    mse = max(0.0, (smm - 2 * som + soo) / n); den = math.sqrt(1 + FM4[p][2] ** 2) * math.sqrt(su2 / n)
    if not den > 0: return None
    mo, mm = so / n, sm / n; sdo = math.sqrt(max(0.0, soo / n - mo * mo)); sdm = math.sqrt(max(0.0, smm / n - mm * mm))
    r = (som / n - mo * mm) / (sdo * sdm) if sdo > 0 and sdm > 0 else 0.0
    biais = mm - mo; crmse = math.sqrt(max(0.0, mse - biais * biais)); dom_r = math.sqrt(max(0.0, 2 * sdo * sdm * (1 - r))) > abs(sdm - sdo)
    return {"mqi": math.sqrt(mse) / den, "x": (-1 if dom_r else 1) * crmse / den, "y": biais / den, "n": n, "rmse": math.sqrt(mse), "biais": biais, "r": r}
def mqi90(v):
    """90e centile des MQI des stations (FAIRMODE : N90 = partie entière de 0,9·Ns, interpolation linéaire)."""
    v = sorted(v); ns = len(v)
    if not ns: return None
    q = 0.9 * ns; k = int(q)
    if k < 1: return v[0]
    if k >= ns: return v[-1]
    return v[k - 1] + (v[k] - v[k - 1]) * (q - k)

def norm(s): return "".join(c for c in unicodedata.normalize("NFD", str(s or "").lower()) if unicodedata.category(c) != "Mn")
def decalage(org):
    o = norm(org)
    return -4 if ("madininair" in o or "gwad" in o) else -3 if "guyane" in o else 4 if "reunion" in o else 3 if ("mayotte" in o or "hawa" in o) else 0
def classe(p, v):
    c = 1
    for b in GRILLE_AEE[p]:
        if v > b: c += 1
    return c
def lire_json(f, defaut):
    try: return json.load(open(f, encoding="utf-8"))
    except Exception: return defaut
def ecrire_json(f, d):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# ---------- Données ----------
REFC = lire_json(os.path.join(REF, "stations_ref.json"), {})
FICHES = lire_json(os.path.join(REF, "fiches_stations.json"), {})
ST = {}
with open(os.path.join(OUT, "archive/lcsqa/stations.csv"), encoding="utf-8", newline="") as fi:
    for r in csv.DictReader(fi, delimiter=";"): ST[r["code"]] = r
V = {}                                    # (code, "AAAA-MM-JJ HH" heure de Paris) -> {polluant: valeur}
for f in sorted(glob.glob(os.path.join(OUT, "archive/lcsqa/*.csv.gz")))[-32:]:
    with gzip.open(f, "rt", encoding="utf-8") as fi:
        rd = csv.reader(fi, delimiter=";"); H = next(rd); ix = {h: i for i, h in enumerate(H)}
        for x in rd:
            code = x[0]; s = ST.get(code)
            if not s: continue
            t = dt.datetime.strptime(x[1], "%Y-%m-%d %H").replace(tzinfo=UTC) - dt.timedelta(hours=decalage(s["organisme"]))
            hk = t.astimezone(PARIS).strftime("%Y-%m-%d %H"); d = V.setdefault((code, hk), {})
            for p in ("NO2", "O3", "PM10", "PM25"):
                v = x[ix[p]]
                if v != "":
                    try: d[p] = max(0.0, float(v))
                    except ValueError: pass
jours_fichiers = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join(OUT, "archive/lcsqa/*.csv.gz")))
# Jour de Paris complet : fichiers du jour et de la veille présents (la première heure de Paris est la dernière du fichier de la veille, en TU), et jour passé
complets = [j for j in jours_fichiers if (dt.date.fromisoformat(j) - dt.timedelta(days=1)).isoformat() in jours_fichiers and dt.date.fromisoformat(j) < AUJ]
print("jours archivés :", jours_fichiers, "; jours de Paris complets :", complets)

def coord(code): c = REFC.get(code); return (c[0], c[1]) if c else None
def km(a, b):
    la = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111.32 * math.cos(la), (a[1] - b[1]) * 110.57)
def est_fond(s): i, f = norm(s["implantation"]), norm(s["influence"]); return ("rural" in i or "urbain" in i) and "fond" in f
def metro(c): return c and -8 < c[0] < 12 and 40 < c[1] < 53
FONDS = [c for c, s in ST.items() if est_fond(s) and coord(c)]
FONDS_URB = [c for c in FONDS if "urbain" in norm(ST[c]["implantation"])]
TRAFIC = [c for c, s in ST.items() if "trafic" in norm(s["influence"]) and "urbain" in norm(s["implantation"]) and coord(c)]

# ---------- Krigeage ordinaire et validation croisée (formule de Dubrule), comme dans la page ----------
def variogramme(x, y, z):
    n = len(z); v = float(np.var(z, ddof=1)) if n > 1 else 0.0
    defaut = [{"nug": 0.1 * v, "sill": 0.9 * v, "range": 100.0}]
    if n < 12 or v <= 0: return defaut
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]; h = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(n, 1); hh = h[iu]; gg = 0.5 * (z[:, None] - z[None, :])[iu] ** 2
    b = np.floor(hh / 15).astype(int); m = b < 20
    g = np.bincount(b[m], weights=gg[m], minlength=20); c = np.bincount(b[m], minlength=20).astype(float)
    ok = c >= 5; centres = (np.arange(20) + 0.5) * 15; cands = []
    for a in (30, 50, 80, 120, 180, 250, 350):
        for nr in (0.02, 0.1, 0.25, 0.4):
            f = nr + (1 - nr) * (1 - np.exp(-3 * centres / a))
            num = np.sum(c[ok] * f[ok] * (g[ok] / c[ok])); den = np.sum(c[ok] * f[ok] ** 2)
            if den <= 0: continue
            tot = num / den
            if tot <= 0: continue
            sse = np.sum(c[ok] * (g[ok] / c[ok] - tot * f[ok]) ** 2)
            cands.append((sse, {"nug": nr * tot, "sill": (1 - nr) * tot, "range": float(a)}))
    if not cands: return defaut
    cands.sort(key=lambda t: t[0]); return [c for _, c in cands[:3]]
def krige_loo(x, y, z):
    n = len(z)
    if n < 4: return None
    best = None
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]; h = np.sqrt(dx * dx + dy * dy)
    for vg in variogramme(x, y, z):
        A = np.ones((n + 1, n + 1)); A[n, n] = 0
        A[:n, :n] = vg["sill"] * np.exp(-3 * h / vg["range"]) + np.eye(n) * vg["nug"]
        try: Q = np.linalg.inv(A)
        except np.linalg.LinAlgError: continue
        w = Q[:, :n] @ z; e = w[:n] / np.diag(Q)[:n]
        r = float(np.sqrt(np.mean(e * e)))
        if best is None or r < best[0]: best = (r, e)
    return None if best is None else best[1]

# ---------- Performances quotidiennes ----------
fperf = os.path.join(OUT, "suivi/performances.json"); PERF = lire_json(fperf, [])
fmj = os.path.join(OUT, "suivi/mqi_jours.json"); MQJ = lire_json(fmj, {})
faits = {d["jour"] for d in PERF if d.get("v") == 4 and d["jour"] in MQJ}               # jours à refaire : antérieurs à FAIRMODE 4.0
journal = lire_json(os.path.join(OUT, "suivi/journal.json"), [])
DOM0 = (2.5, 46.5); KX = 111.32 * math.cos(math.radians(DOM0[1]))
FM = [c for c in FONDS if metro(coord(c))]
for jour in [j for j in complets if j not in faits][-5:]:
    t1 = time.time(); heures = ["%s %02d" % (jour, h) for h in range(24)]
    cnt = {hk: sum(1 for c in FM if V.get((c, hk))) for hk in heures}; mx = max(cnt.values()) if cnt else 0
    hs = [hk for hk in heures if cnt[hk] >= max(20, 0.4 * mx)]
    res = {"jour": jour, "heures": len(hs), "pol": {}, "v": 4}; MQJ[jour] = {}
    for p in POLS:
        series = {}; se = sb = n = ok = 0
        for hk in hs:
            cs = [c for c in FM if p in V.get((c, hk), {})]
            if len(cs) < 12: continue
            x = np.array([(coord(c)[0] - DOM0[0]) * KX for c in cs]); y = np.array([(coord(c)[1] - DOM0[1]) * 110.57 for c in cs])
            z = np.array([V[(c, hk)][p] for c in cs]); e = krige_loo(x, y, z)
            if e is None: continue
            for c, o, ei in zip(cs, z, e):
                est = max(0.0, o - ei); se += (est - o) ** 2; sb += est - o; n += 1
                if classe(p, o) == classe(p, est): ok += 1
                series.setdefault(c, []).append((int(hk[-2:]), float(o), float(est)))
        MQJ[jour][p] = {}
        for c, serie in series.items():
            S = sommes(p, serie)
            if S: MQJ[jour][p][c] = [round(x, 3) for x in S]
        res["pol"][p] = {"rmse": round(math.sqrt(se / n), 2) if n else None, "biais": round(sb / n, 2) if n else None, "n": n,
                         "nst": len(series), "classe": round(ok / n, 3) if n else None}
    jj = [x for x in journal if str(x.get("t", "")).startswith(jour)]
    res["copies"] = {"n": len(jj), "complets": sum(1 for x in jj if x.get("lcsqa") == x.get("trafic") == x.get("arome") == "success"),
                     "lcsqa": sum(1 for x in jj if x.get("lcsqa") == "success")} if jj else None
    PERF.append(res); print("performances", jour, ":", len(hs), "heures,", round(time.time() - t1, 1), "s")
# MQI sur fenêtres glissantes de 7 et 30 jours, recalculé pour chaque jour à partir des sommes quotidiennes par station
MQJ = {j: MQJ[j] for j in sorted(MQJ)[-30:]}
def fenetre(fin, w, p, details=False):
    js = [j for j in sorted(MQJ) if j <= fin][-w:]
    js = [j for j in js if (dt.date.fromisoformat(fin) - dt.date.fromisoformat(j)).days < w]
    tot = {}
    for j in js:
        for c, S in MQJ[j].get(p, {}).items():
            t = tot.setdefault(c, [0.0] * 7)
            for k in range(7): t[k] += S[k]
    if not tot: return None
    nmax = max(t[0] for t in tot.values()); seuil = max(1, 0.75 * nmax)                  # couverture : 75 % de la station la mieux suivie
    st = {c: mqi_station(p, t) for c, t in tot.items() if t[0] >= seuil}
    st = {c: v for c, v in st.items() if v}
    if not st: return None
    out = {"mqi90": round(mqi90([v["mqi"] for v in st.values()]), 3), "ns": len(st), "nd": len(js)}
    if details: out["pts"] = [[c, round(v["x"], 3), round(v["y"], 3), round(v["mqi"], 3), v["n"], round(v["rmse"], 2), round(v["biais"], 2), round(v["r"], 3)] for c, v in sorted(st.items())]
    return out
for d in PERF:
    if d.get("v") != 4 or d["jour"] not in MQJ: continue
    for p in POLS:
        q = d["pol"].get(p)
        if q is None: continue
        f7, f30 = fenetre(d["jour"], 7, p), fenetre(d["jour"], 30, p)
        q["mqi90"] = f7["mqi90"] if f7 else None; q["nmqi"] = f7["ns"] if f7 else 0; q["mqi90_30"] = f30["mqi90"] if f30 else None
PERF = sorted(PERF, key=lambda d: d["jour"])[-90:]
ecrire_json(fperf, PERF); ecrire_json(fmj, MQJ)
if MQJ:
    fin = max(MQJ)
    ecrire_json(os.path.join(OUT, "suivi/mqi_fenetres.json"), {"version": "FAIRMODE 4.0 (mai 2026, tableau 1)", "jour": fin,
        "periodes": {"NO2": "horaire", "O3": "maximum journalier sur 8 h", "PM10": "moyenne journalière", "PM25": "moyenne journalière"},
        "parametres": {p: {"LV": FM4[p][0], "Ur": FM4[p][1], "beta": FM4[p][2], "U0": FM4[p][3]} for p in POLS},
        "7": {p: fenetre(fin, 7, p, True) for p in POLS}, "30": {p: fenetre(fin, 30, p, True) for p in POLS},
        "pts": "code, x (CRMSE normé, signé), y (biais normé), MQI, nombre de valeurs, erreur, biais, corrélation"})

# ---------- Calage sur l'historique (30 derniers jours complets) ----------
jours = complets[-30:]
paires = []
for t in TRAFIC:
    ct = coord(t); near = sorted((km(ct, coord(f)), f) for f in FONDS_URB if km(ct, coord(f)) <= 10)[:3]
    if near: paires.append((t, [f for _, f in near]))
def fond_moyen(fs, hk, p):
    v = [V[(f, hk)][p] for f in fs if p in V.get((f, hk), {})]
    return sum(v) / len(v) if v else None
# Vent et pluie AROME aux stations trafic (fichiers du jour gardés 30 jours, heure de Paris)
AR = {}
meteo = lire_json(os.path.join(OUT, "arome/meteo.json"), None)
if meteo:
    g = meteo["grille"]; N = g["ni"] * g["nj"]
    for j in jours + [AUJ.isoformat()]:
        f = os.path.join(OUT, "arome", j + ".bin")
        if os.path.exists(f) and os.path.getsize(f) == 24 * 3 * N:
            AR[j] = np.frombuffer(open(f, "rb").read(), dtype=np.int8).reshape(24, 3, N)
def arome(c, hk):
    a = AR.get(hk[:10])
    if a is None: return None
    g = meteo["grille"]; fx = (c[0] - g["lon0"]) / g["pas"] - 0.5; fy = (g["lat0"] - c[1]) / g["pas"] - 0.5
    if fx < 0 or fy < 0 or fx > g["ni"] - 1 or fy > g["nj"] - 1: return None
    i, j = int(fx), int(fy); ax, ay = fx - i, fy - j; i1, j1 = min(g["ni"] - 1, i + 1), min(g["nj"] - 1, j + 1); h = int(hk[11:13])
    ks = [j * g["ni"] + i, j * g["ni"] + i1, j1 * g["ni"] + i, j1 * g["ni"] + i1]
    if any(a[h, 0, k] == -128 for k in ks): return None
    bil = lambda o: ((a[h, o, ks[0]] * (1 - ax) + a[h, o, ks[1]] * ax) * (1 - ay) + (a[h, o, ks[2]] * (1 - ax) + a[h, o, ks[3]] * ax) * ay) * g["echelle_vent"]
    u, v = bil(0), bil(1); pr = int(np.uint8(a[h, 2, int(round(fy)) * g["ni"] + int(round(fx))]))
    return math.hypot(u, v), (0.0 if pr == 255 else pr * g["echelle_pluie"])

CAL = {"maj": dt.datetime.now(PARIS).isoformat(timespec="seconds"), "jours": jours, "nj": len(jours), "paires": len(paires),
       "D": {}, "R": {}, "n": {}, "g": {"ouvre": {}, "weekend": {}}, "vent": None, "pluie": None, "fiches": None}
obs = {p: [] for p in ("NO2", "PM10", "PM25")}           # (heure, type de jour, T, F, code)
for j in jours:
    we = dt.date.fromisoformat(j).weekday() >= 5
    for h in range(24):
        hk = "%s %02d" % (j, h)
        for t, fs in paires:
            dv = V.get((t, hk))
            if not dv: continue
            for p in obs:
                if p not in dv: continue
                fb = fond_moyen(fs, hk, p)
                if fb is not None: obs[p].append((hk, we, dv[p], fb, t))
for p, L in obs.items():
    if len(L) < 60: continue
    sT = sum(o[2] for o in L); sF = sum(o[3] for o in L); D = (sT - sF) / len(L)
    CAL["D"][p] = round(max(0.0, D), 2); CAL["R"][p] = round(sT / sF, 3) if sF > 0 else None; CAL["n"][p] = len(L)
    if D > 0.2:
        for typ, w in (("ouvre", False), ("weekend", True)):
            prof = {}
            for h in range(24):
                d = [o[2] - o[3] for o in L if o[1] == w and int(o[0][11:13]) == h]
                if len(d) >= 30: prof[h] = round(max(0.2, min(2.5, (sum(d) / len(d)) / D)), 3)
            if prof: CAL["g"][typ][p] = prof
# Exposant du vent (NO₂) : incrément moyen par quart de vitesse de vent, droite de régression en log-log (comme dans la page)
if AR:
    ov = []
    for hk, we, T, F, t in obs["NO2"]:
        a = arome(coord(t), hk)
        if a: ov.append((max(0.3, a[0]), T, F))
    if len(ov) >= 200:
        ov.sort(); q = len(ov) // 4; X = []; Y = []
        for b in range(4):
            gp = ov[b * q:(len(ov) if b == 3 else (b + 1) * q)]; dg = sum(o[1] - o[2] for o in gp) / len(gp)
            if dg > 0.2: X.append(math.log(sum(o[0] for o in gp) / len(gp))); Y.append(math.log(dg))
        if len(X) >= 3:
            pente = -float(np.polyfit(X, Y, 1)[0])
            CAL["vent"] = {"a": round(max(0.0, min(1.0, pente)), 3), "pente": round(pente, 3), "n": len(ov)}
# Pluie (particules) : Δ ≈ D · g(h) · (1 − min(0,5 ; k · P)), k ajusté sur une grille ; 100 heures-couples pluvieuses au moins
if AR and "PM10" in CAL["D"] and CAL["D"]["PM10"] > 0.2:
    D = CAL["D"]["PM10"]; op = []
    for hk, we, T, F, t in obs["PM10"]:
        a = arome(coord(t), hk)
        if a is None: continue
        gg = CAL["g"]["weekend" if we else "ouvre"].get("PM10", {}).get(int(hk[11:13]), 1.0)
        op.append((a[1], T - F, D * gg))
    npl = sum(1 for o in op if o[0] >= 0.2)
    if npl >= 100:
        best = None
        for k10 in range(0, 21):
            k = k10 / 20; e = sum((o[2] * (1 - min(0.5, k * o[0])) - o[1]) ** 2 for o in op)
            if best is None or e < best[0]: best = (e, k)
        CAL["pluie"] = {"k": best[1], "n": len(op), "npluie": npl}
    else: CAL["pluie"] = {"k": None, "n": len(op), "npluie": npl}
# Trafic et confinement sur les stations documentées : Δ = D · (TMJA / 30 000)^a · (1 + c·H/L) / (1 + c/3)
pts = []
for code, f in FICHES.items():
    if not f.get("tmja"): continue
    d = [o[2] - o[3] for o in obs["NO2"] if o[4] == code]
    if len(d) >= 24: pts.append({"code": code, "q": f["tmja"], "A": f["A"] if f.get("A") is not None else 1 / 3, "d": sum(d) / len(d), "nh": len(d)})
def pred(p, a, c): return (p["q"] / 30000) ** a * (1 + c * p["A"]) / (1 + c / 3)
def ajuste(L, grille):
    best = None
    for a, c in grille:
        m = [pred(p, a, c) for p in L]; den = sum(x * x for x in m)
        if den <= 0: continue
        D = sum(x * p["d"] for x, p in zip(m, L)) / den; e = sum((D * x - p["d"]) ** 2 for x, p in zip(m, L))
        if best is None or e < best[0]: best = (e, D, a, c)
    return best
G = [(ia / 20, ic / 10) for ia in range(21) for ic in range(16)]
def loo(grille):
    e = 0
    for i, p in enumerate(pts):
        b = ajuste(pts[:i] + pts[i + 1:], grille); e += (b[1] * pred(p, b[2], b[3]) - p["d"]) ** 2
    return math.sqrt(e / len(pts))
if len(pts) >= 12:
    b = ajuste(pts, G)
    CAL["fiches"] = {"a": b[2], "c": b[3], "D": round(b[1], 2), "n": len(pts),
                     "loo": {"constant": round(loo([(0, 0)]), 2), "ancien": round(loo([(1, 1)]), 2), "cale": round(loo(G), 2)},
                     "pts": [{"code": p["code"], "d": round(p["d"], 2), "m": round(b[1] * pred(p, b[2], b[3]), 2), "nh": p["nh"]} for p in pts]}
ecrire_json(os.path.join(OUT, "calage/calage.json"), CAL)
print("calage :", len(jours), "jours,", len(paires), "couples ; D", CAL["D"], "; vent", CAL["vent"], "; pluie", CAL["pluie"],
      "; fiches", {k: v for k, v in (CAL["fiches"] or {}).items() if k != "pts"}, "; durée", round(time.time() - T0, 1), "s")

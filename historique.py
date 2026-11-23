# IATMO : calage sur l'historique et performances quotidiennes, à partir de l'archive de la branche « donnees ».
# Lancé une fois par jour par la tâche GitHub (.github/workflows/copie_lcsqa.yml), après la copie.
#   python3 historique.py <dossier de la branche donnees> <dossier contenant stations_ref.json et fiches_stations.json>
# Écrit :
#   calage/calage.json      : incrément trafic − fond, profil horaire (jours ouvrés, week-end), exposant du vent, effet de la pluie,
#                             exposant du trafic et confinement calés sur les stations documentées, sur les 30 derniers jours complets ;
#                             P5 (clé « p5 ») : incrément de NOx et son profil (clés D, g « NOX »), exposant du vent et coefficient de
#                             température des NOx ; conversion en NO2 selon l'ozone de fond (valeurs pluriannuelles, estimation sur 30 jours en suivi) ;
#   suivi/performances.json : pour chaque jour complet, validation croisée du krigeage ordinaire (station retirée tour à tour) par
#                             polluant, taux de bonne classe, MQI de FAIRMODE sur 7 et 30 jours glissants, et bilan des copies (90 jours) ;
#   suivi/mqi_jours.json    : sommes quotidiennes par station et par polluant pour le MQI (30 jours) ;
#   suivi/mqi_fenetres.json : MQI par station et MQI du 90e centile sur les 7 et 30 derniers jours (diagramme cible de la page).
#   suivi/calage_jours.json : progression du calage, jour après jour (400 jours) : paramètres calés sur les 30 jours précédant
#                             chaque jour, intervalles à 90 % et test à l'aveugle sur ce jour (fond seul, paramètres d'origine, calés).
#                             Correction apprise (gradient boosting et perceptron, scikit-learn) des résidus aux stations trafic : test à l'aveugle
#                             et validation croisée par agglomération écartée (clé « ml »), pour les derniers jours.
# MQI : guide FAIRMODE version 4.1 (septembre 2026 ; tableau 1 inchangé depuis la 4.0 de mai 2026 ; directive 2024/2881) ; NO₂ horaire, O₃ maximum journalier sur 8 h,
#       PM₁₀ et PM₂,₅ moyenne journalière. Paramètres à revoir avec la mise à jour annoncée pour fin 2026.
# Heures : celles des fichiers du LCSQA (temps universel en métropole, heure locale dans les DROM) sont converties en heure de Paris.
import csv, gzip, glob, hashlib, json, math, os, sys, time, unicodedata, warnings, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np

T0 = time.time()
OUT, REF = sys.argv[1], sys.argv[2]
PARIS = ZoneInfo("Europe/Paris"); UTC = dt.timezone.utc
AUJ = dt.date.fromisoformat(os.environ["IATMO_JOUR"]) if os.environ.get("IATMO_JOUR") else dt.datetime.now(PARIS).date()   # IATMO_JOUR : essais
POLS = ["NO2", "O3", "PM10", "PM25"]
GRILLE_AEE = {"NO2": [10, 25, 60, 100, 150], "O3": [60, 100, 120, 160, 180], "PM10": [15, 45, 120, 195, 270], "PM25": [5, 15, 50, 90, 140]}
# FAIRMODE 4.1, tableau 1 : valeur limite LV (µg/m³), U_r,fix, β court terme, plancher U∅ (µg/m³), période d'évaluation
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
RATTRAPAGE = int(os.environ.get("IATMO_RATTRAPAGE", 7))   # jours de test à l'aveugle recalculés au plus par passage (suivi/calage_jours.json)
for f in sorted(glob.glob(os.path.join(OUT, "archive/lcsqa/*.csv.gz")))[-(32 + RATTRAPAGE):]:
    with gzip.open(f, "rt", encoding="utf-8") as fi:
        rd = csv.reader(fi, delimiter=";"); H = next(rd); ix = {h: i for i, h in enumerate(H)}
        for x in rd:
            code = x[0]; s = ST.get(code)
            if not s: continue
            t = dt.datetime.strptime(x[1], "%Y-%m-%d %H").replace(tzinfo=UTC) - dt.timedelta(hours=decalage(s["organisme"]))
            hk = t.astimezone(PARIS).strftime("%Y-%m-%d %H"); d = V.setdefault((code, hk), {})
            for p in ("NO2", "O3", "PM10", "PM25", "NO", "NOX"):
                v = x[ix[p]] if p in ix else ""
                if v != "":
                    try: d[p] = max(0.0, float(v))
                    except ValueError: pass
            if "NOX" not in d and "NO" in d and "NO2" in d: d["NOX"] = d["NO2"] + 46.0055 / 30.0061 * d["NO"]   # NOx en équivalent NO2
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
faits = {d["jour"] for d in PERF if d.get("v") == 4 and d["jour"] in MQJ}               # jours à refaire : antérieurs au format FAIRMODE 4 (v = 4)
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
    PERF = [d for d in PERF if d["jour"] != jour] + [res]; print("performances", jour, ":", len(hs), "heures,", round(time.time() - t1, 1), "s")
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
uniq = {}                                                                                  # un seul enregistrement par jour : le format le plus récent l'emporte
for d in sorted(PERF, key=lambda d: (d["jour"], d.get("v", 0))): uniq[d["jour"]] = d
PERF = [uniq[j] for j in sorted(uniq)][-90:]
ecrire_json(fperf, PERF); ecrire_json(fmj, MQJ)
if MQJ:
    fin = max(MQJ)
    ecrire_json(os.path.join(OUT, "suivi/mqi_fenetres.json"), {"version": "FAIRMODE 4.1 (septembre 2026, tableau 1)", "jour": fin,
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
# Vent, pluie, température AROME aux stations trafic (fichiers du jour gardés 30 jours, heure de Paris)
AR = {}; TA = {}
meteo = lire_json(os.path.join(OUT, "arome/meteo.json"), None)
if meteo:
    g = meteo["grille"]; N = g["ni"] * g["nj"]
    for j in complets[-(30 + RATTRAPAGE):] + [AUJ.isoformat()]:
        f = os.path.join(OUT, "arome", j + ".bin")
        if os.path.exists(f) and os.path.getsize(f) == 24 * 3 * N:
            AR[j] = np.frombuffer(open(f, "rb").read(), dtype=np.int8).reshape(24, 3, N)
        f = os.path.join(OUT, "arome", "ta-" + j + ".bin")                    # température (demi-degrés) et humidité, depuis le 24/11/2026
        if os.path.exists(f) and os.path.getsize(f) == 24 * 2 * N:
            TA[j] = np.frombuffer(open(f, "rb").read(), dtype=np.int8).reshape(24, 2, N)
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
    t2 = None; b = TA.get(hk[:10])
    if b is not None:
        tv = [int(b[h, 0, k]) for k in ks]
        if -128 not in tv: t2 = ((tv[0] * (1 - ax) + tv[1] * ax) * (1 - ay) + (tv[2] * (1 - ax) + tv[3] * ax) * ay) * (meteo.get("ta") or {}).get("echelle_temperature", 0.5)
    return math.hypot(u, v), (0.0 if pr == 255 else pr * g["echelle_pluie"]), t2

# P5 : incrément de NOx = D · g(h) · vent · température · pluie, converti en NO2 selon l'ozone de fond :
# ΔNO2 = ΔNOx · [f0 + (1 − f0) · (1 − exp(−O3 / O3*))]. Valeurs par défaut : historique Geod'air 2023-2025 et AROME HD (calage 2023-2024,
# test à l'aveugle 2025, confirmé sur 2026) ; la carte utilise les mêmes.
P5_DEF = {"vent": 0.2, "temp": 0.02, "pluie": 0.1, "f0": 0.10, "o3s": 185.0, "source": "historique Geod'air 2023-2025, AROME HD (Open-Meteo)"}
def p5_conv(o3): return P5_DEF["f0"] + (1 - P5_DEF["f0"]) * (1 - math.exp(-max(0.0, o3) / P5_DEF["o3s"]))
def p5_fac(p5, u, P, T2, cale=True):
    # Vent et température rapportés à leurs valeurs moyennes sur la fenêtre de calage (u0, t0), puisque l'incrément D_NOx est calé sur
    # cette même fenêtre ; exposant et coefficient calés si cale est vrai et s'ils existent, valeurs pluriannuelles sinon.
    v = (p5 or {}).get("vent") or {}; te = (p5 or {}).get("temp") or {}
    a = v["a"] if cale and v.get("a") is not None else P5_DEF["vent"]; u0 = v.get("u0") or 3.0
    g = te["g"] if cale and te.get("g") is not None else P5_DEF["temp"]; t0 = te.get("t0", 12.0)
    fv = 1.0 if u is None else max(0.6, min(1.6, (u0 / max(u, 0.3)) ** a))
    ft = 1.0 if T2 is None else max(0.6, min(1.6, math.exp(-g * (T2 - t0))))
    fp = 1.0 if P is None else math.exp(-P5_DEF["pluie"] * min(P, 5.0))
    return fv * ft * fp
def pente_vent(ov):
    """Exposant du vent (NO₂) : incrément moyen par quart de vitesse de vent, droite de régression en log-log (comme dans la page)."""
    if len(ov) < 200: return None
    ov = sorted(ov); q = len(ov) // 4; X = []; Y = []
    for b in range(4):
        gp = ov[b * q:(len(ov) if b == 3 else (b + 1) * q)]; dg = sum(o[1] - o[2] for o in gp) / len(gp)
        if dg > 0.2: X.append(math.log(sum(o[0] for o in gp) / len(gp))); Y.append(math.log(dg))
    return -float(np.polyfit(X, Y, 1)[0]) if len(X) >= 3 else None
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
def centiles(v, d=3):
    v = sorted(x for x in v if x is not None)
    return [round(v[int(0.05 * (len(v) - 1))], d), round(v[int(round(0.95 * (len(v) - 1)))], d)] if len(v) >= 10 else None

def caler(jours, boot=False):
    """Calage sur les jours donnés. boot=True : intervalles à 90 % par rééchantillonnage (des jours, ou des stations documentées)."""
    CAL = {"maj": dt.datetime.now(PARIS).isoformat(timespec="seconds"), "jours": jours, "nj": len(jours), "paires": len(paires),
           "D": {}, "R": {}, "n": {}, "g": {"ouvre": {}, "weekend": {}}, "vent": None, "pluie": None, "fiches": None}
    obs = {p: [] for p in ("NO2", "PM10", "PM25", "NOX")}    # (heure, type de jour, T, F, code)
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
    CAL["nj_obs"] = len({o[0][:10] for L in obs.values() for o in L})          # jours de la fenêtre qui ont des mesures
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
    # Vent (NO₂) ; u0 : vitesse de référence (moyenne géométrique de la fenêtre), pour le test à l'aveugle
    ovj = {}
    if AR:
        for hk, we, T, F, t in obs["NO2"]:
            a = arome(coord(t), hk)
            if a: ovj.setdefault(hk[:10], []).append((max(0.3, a[0]), T, F))
        ov = [o for L in ovj.values() for o in L]; pente = pente_vent(ov)
        if pente is not None:
            CAL["vent"] = {"a": round(max(0.0, min(1.0, pente)), 3), "pente": round(pente, 3), "n": len(ov),
                           "u0": round(math.exp(sum(math.log(o[0]) for o in ov) / len(ov)), 2)}
    # P5 : exposant du vent et coefficient de température des NOx (la pluie garde sa valeur pluriannuelle) ; chimie estimée sur la
    # fenêtre pour le suivi seulement (la carte garde f0 et O3* pluriannuels, mieux contraints que sur 30 jours)
    p5 = {"defauts": P5_DEF, "vent": None, "temp": None, "chimie_30j": None}
    if AR and CAL["D"].get("NOX"):
        D = CAL["D"]["NOX"]; ovx = []; ot = []
        for hk, we, T, F, t in obs["NOX"]:
            a = arome(coord(t), hk)
            if not a: continue
            ovx.append((max(0.3, a[0]), T, F))
            if a[2] is not None:
                gg = CAL["g"]["weekend" if we else "ouvre"].get("NOX", {}).get(int(hk[11:13]), 1.0); ot.append((a[2], T - F, D * gg, max(0.3, a[0])))
        pe = pente_vent(ovx) if len(ovx) >= 5000 else None                 # 5 000 heures-couples au moins (environ 4 jours) ; sinon valeur pluriannuelle
        if ovx: p5["vent"] = {"a": None if pe is None else round(max(0.0, min(0.6, pe)), 3), "n": len(ovx), "u0": round(math.exp(sum(math.log(o[0]) for o in ovx) / len(ovx)), 2)}
        # Température : ajustée avec le facteur du vent (le doux et le venteux vont souvent ensemble en hiver), et seulement si la fenêtre
        # couvre au moins 8 °C entre ses 5e et 95e centiles ; sinon valeur pluriannuelle.
        etendue = float(np.percentile([o[0] for o in ot], 95) - np.percentile([o[0] for o in ot], 5)) if ot else 0.0
        t0 = round(sum(o[0] for o in ot) / len(ot), 1) if ot else None    # température moyenne de la fenêtre : référence du facteur
        if len(ot) >= 500 and etendue >= 8:
            av = (p5["vent"] or {}).get("a"); av = P5_DEF["vent"] if av is None else av; u0v = (p5["vent"] or {}).get("u0") or 3.0
            ot = [(tt, d, m * max(0.6, min(1.6, (u0v / uu) ** av))) for tt, d, m, uu in ot]
            err = [(sum((m * math.exp(-k / 200 * (tt - t0)) - d) ** 2 for tt, d, m in ot), k / 200) for k in range(17)]
            p5["temp"] = {"g": min(err)[1], "n": len(ot), "t0": t0, "etendue": round(etendue, 1)}
        else: p5["temp"] = {"g": None, "n": len(ot), "t0": t0, "etendue": round(etendue, 1)}
        if t0 is None: del p5["temp"]["t0"]
    oc = []
    for j in jours:
        for h in range(24):
            hk = "%s %02d" % (j, h)
            for t, fs in paires:
                dv = V.get((t, hk))
                if not dv or "NOX" not in dv or "NO2" not in dv: continue
                fx, f2, o3 = fond_moyen(fs, hk, "NOX"), fond_moyen(fs, hk, "NO2"), fond_moyen(fs, hk, "O3")
                if None not in (fx, f2, o3) and dv["NOX"] - fx > 10: oc.append((dv["NOX"] - fx, dv["NO2"] - f2, o3))
    if len(oc) >= 2000:
        X = np.array(oc); best = None
        for f0 in np.arange(0.0, 0.31, 0.02):
            for o3s in np.arange(20, 301, 10):
                e = float(np.mean((X[:, 0] * (f0 + (1 - f0) * (1 - np.exp(-X[:, 2] / o3s))) - X[:, 1]) ** 2))
                if best is None or e < best[0]: best = (e, f0, o3s)
        p5["chimie_30j"] = {"f0": round(float(best[1]), 2), "o3s": int(best[2]), "n": len(oc)}
    CAL["p5"] = p5
    # Pluie (particules) : Δ ≈ D · g(h) · (1 − min(0,5 ; k · P)), k ajusté sur une grille ; 100 heures-couples pluvieuses au moins
    Ej = {}
    if AR and "PM10" in CAL["D"] and CAL["D"]["PM10"] > 0.2:
        D = CAL["D"]["PM10"]; op = []
        for hk, we, T, F, t in obs["PM10"]:
            a = arome(coord(t), hk)
            if a is None: continue
            gg = CAL["g"]["weekend" if we else "ouvre"].get("PM10", {}).get(int(hk[11:13]), 1.0)
            op.append((a[1], T - F, D * gg, hk[:10]))
        npl = sum(1 for o in op if o[0] >= 0.2)
        for o in op:
            e = Ej.setdefault(o[3], [0.0] * 21)
            for k10 in range(21): e[k10] += (o[2] * (1 - min(0.5, k10 / 20 * o[0])) - o[1]) ** 2
        if npl >= 100:
            tot = [sum(e[k] for e in Ej.values()) for k in range(21)]
            CAL["pluie"] = {"k": tot.index(min(tot)) / 20, "n": len(op), "npluie": npl}
        else: CAL["pluie"] = {"k": None, "n": len(op), "npluie": npl}
    # Trafic et confinement sur les stations documentées : Δ = D · (TMJA / 30 000)^a · (1 + c·H/L) / (1 + c/3)
    pts = []
    for code, f in FICHES.items():
        if not f.get("tmja"): continue
        d = [o[2] - o[3] for o in obs["NO2"] if o[4] == code]
        if len(d) >= 24: pts.append({"code": code, "q": f["tmja"], "A": f["A"] if f.get("A") is not None else 1 / 3, "d": sum(d) / len(d), "nh": len(d)})
    CAL["nst"] = len(pts)
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
    if not boot: return CAL, None
    # Intervalles à 90 % : jours tirés avec remise (incrément, vent, pluie), stations documentées tirées avec remise (trafic, confinement)
    import random
    rg = random.Random(jours[-1] if jours else "0"); IC = {"D": {}}
    for p, L in obs.items():
        if p not in CAL["D"]: continue
        dj = {}
        for o in L: s = dj.setdefault(o[0][:10], [0.0, 0]); s[0] += o[2] - o[3]; s[1] += 1
        ks = list(dj); v = []
        for _ in range(200):
            tir = [dj[ks[rg.randrange(len(ks))]] for _ in ks]; n = sum(s[1] for s in tir)
            if n: v.append(max(0.0, sum(s[0] for s in tir) / n))
        IC["D"][p] = centiles(v, 2)
    if CAL["vent"]:
        ks = list(ovj); v = []
        for _ in range(40):
            pe = pente_vent([o for _ in ks for o in ovj[ks[rg.randrange(len(ks))]]])
            if pe is not None: v.append(max(0.0, min(1.0, pe)))
        IC["vent"] = centiles(v)
    if CAL["pluie"] and CAL["pluie"]["k"] is not None:
        ks = list(Ej); v = []
        for _ in range(200):
            tir = [Ej[ks[rg.randrange(len(ks))]] for _ in ks]; tot = [sum(e[k] for e in tir) for k in range(21)]
            v.append(tot.index(min(tot)) / 20)
        IC["pluie"] = centiles(v, 2)
    if CAL["fiches"]:
        va = []; vc = []
        for _ in range(60):
            b = ajuste([pts[rg.randrange(len(pts))] for _ in pts], G); va.append(b[2]); vc.append(b[3])
        IC["a"] = centiles(va, 2); IC["c"] = centiles(vc, 2)
    return CAL, IC

CAL, _ = caler(jours)
ecrire_json(os.path.join(OUT, "calage/calage.json"), CAL)
print("calage :", len(jours), "jours,", len(paires), "couples ; D", CAL["D"], "; vent", CAL["vent"], "; pluie", CAL["pluie"],
      "; fiches", {k: v for k, v in (CAL["fiches"] or {}).items() if k != "pts"}, "; durée", round(time.time() - T0, 1), "s")

# ---------- Progression du calage : test à l'aveugle, jour après jour (suivi/calage_jours.json) ----------
# Pour chaque jour complet J, paramètres calés sur les 30 jours précédant J, appliqués aux couples trafic / fond de J, comparés
# au fond seul (moyenne des stations de fond appariées) et aux paramètres d'origine de la page. Comme la carte sans mesures du jour :
# vent, pluie, trafic et confinement calés seulement à partir de HIST_MIN jours d'historique.
HIST_MIN = 7
GREF = {1: 0.33, 2: 0.32, 3: 0.37, 4: 0.55, 5: 0.88, 6: 1.34, 7: 1.55, 8: 1.61, 9: 1.33, 10: 1.40, 11: 1.39}   # référence du 16/11, jours ouvrés
def f_vent_ref(u): return max(0.6, min(1.6, math.sqrt(3 / max(u, 0.3))))                              # dilution, référence 3 m/s
P5PRED = {}                                                  # (station, jour, heure) -> (incrément de NO2 prévu par P5, vent et température AROME lus)
def lignes(jours_l, C):
    """Couples station trafic / fond, heure par heure : (polluant, station, jour, heure, week-end, vent, pluie, fond, incrément
    d'origine, incrément calé, mesure à la station trafic). Même calcul que la carte sans mesures du jour."""
    hist = C["nj_obs"] >= HIST_MIN; out = []
    for J in jours_l:
        we = dt.date.fromisoformat(J).weekday() >= 5
        for t, fs in paires:
            fi = FICHES.get(t) if hist and C["fiches"] else None
            if fi and not fi.get("tmja"): fi = None
            for h in range(24):
                hk = "%s %02d" % (J, h); dv = V.get((t, hk))
                if not dv: continue
                met = arome(coord(t), hk); u = met[0] if met else None; P = met[1] if met else None; T2 = met[2] if met else None
                Pp = P or 0.0; fvr = f_vent_ref(u) if u is not None else 1.0
                fv = fvr
                if u is not None and hist and C["vent"]: fv = max(0.6, min(1.6, (C["vent"]["u0"] / max(u, 0.3)) ** C["vent"]["a"]))
                k = C["pluie"]["k"] if hist and C["pluie"] and C["pluie"]["k"] is not None and C["pluie"]["npluie"] >= 100 else 0.25
                for p in ("NO2", "PM10", "PM25"):
                    if p not in dv or C["D"].get(p) is None: continue
                    F = fond_moyen(fs, hk, p)
                    if F is None: continue
                    if p == "NO2": dref = 8.8 * max(0.3, min(2.5, (1.0 if we else GREF.get(h, 1.0)) * fvr))
                    else: dref = 0.3 * F * max(0.3, min(2.5, fvr * (1 - min(0.5, 0.25 * Pp))))
                    fac = C["g"]["weekend" if we else "ouvre"].get(p, {}).get(h, 1.0) * fv * (1 - min(0.5, k * Pp) if p == "PM10" else 1)   # pluie : PM10 seulement, comme la carte
                    base = C["D"][p]
                    if p == "NO2" and fi: base = C["fiches"]["D"] * pred({"q": fi["tmja"], "A": fi["A"] if fi.get("A") is not None else 1 / 3}, C["fiches"]["a"], C["fiches"]["c"])
                    out.append((p, t, J, h, we, u, P, F, dref, base * max(0.3, min(2.5, fac)), dv[p]))
                    if p == "NO2" and C["D"].get("NOX") and C["D"].get("NO2"):
                        o3b = fond_moyen(fs, hk, "O3")
                        if o3b is not None:
                            gx = C["g"]["weekend" if we else "ouvre"].get("NOX", {}).get(h, 1.0)
                            dnox = C["D"]["NOX"] * (base / C["D"]["NO2"]) * max(0.3, min(2.5, gx * p5_fac(C.get("p5"), u, P, T2, cale=hist)))
                            P5PRED[(t, J, h)] = (dnox * p5_conv(o3b), u is not None and T2 is not None)
    return out
def tester(J, C):
    acc = {p: [0, 0.0, 0.0, 0.0, 0.0] for p in ("NO2", "PM10", "PM25")}     # n, Σ(fond seul)², Σ(origine)², Σ(calé)², Σ biais calé
    p5 = [0, 0.0, 0.0, 0.0, 0.0, 0]                                         # NO2, heures avec ozone de fond : n, Σ(fond)², Σ(calé)², Σ(P5)², Σ biais P5, heures avec météo
    for p, t, Jj, h, we, u, P, F, dref, dc, T in lignes([J], C):
        a = acc[p]; a[0] += 1; a[1] += (F - T) ** 2; a[2] += (F + dref - T) ** 2; a[3] += (F + dc - T) ** 2; a[4] += F + dc - T
        q = P5PRED.get((t, Jj, h)) if p == "NO2" else None
        if q is not None:
            q, m = q; p5[0] += 1; p5[1] += (F - T) ** 2; p5[2] += (F + dc - T) ** 2; p5[3] += (F + q - T) ** 2; p5[4] += F + q - T; p5[5] += m
    r = lambda s, n: round(math.sqrt(s / n), 2)
    out = {p: {"n": a[0], "fond": r(a[1], a[0]), "ref": r(a[2], a[0]), "cale": r(a[3], a[0]), "biais": round(a[4] / a[0], 2)} for p, a in acc.items() if a[0] >= 50}
    if "NO2" in out and p5[0] >= 50:
        out["NO2"]["p5"] = {"n": p5[0], "fond": r(p5[1], p5[0]), "cale": r(p5[2], p5[0]), "p5": r(p5[3], p5[0]), "biais": round(p5[4] / p5[0], 2), "meteo": round(p5[5] / p5[0], 2)}
    return out

# ---------- Correction apprise des résidus (gradient boosting), évaluée à l'aveugle et par agglomération écartée ----------
# Le modèle paramétrique reste la base : un modèle de gradient boosting apprend, sur les 30 jours de la fenêtre, l'écart
# résiduel mesure − (fond + incrément calé) aux stations trafic, à partir de variables disponibles aussi sur la carte.
# Deux candidats comparés : gradient boosting (avec contraintes de monotonie) et perceptron multicouche (moyenne de graines).
# Évaluation : (1) test à l'aveugle sur le jour J, toutes stations ; (2) NO₂ : validation croisée en 5 groupes de zones
# administratives de surveillance (ZAS) : la correction est apprise sans les stations du groupe, puis appliquée à elles le jour J.
try:
    import sklearn; from sklearn.ensemble import HistGradientBoostingRegressor; from sklearn.neural_network import MLPRegressor; SKL = sklearn.__version__
except Exception: SKL = None
ML_MAX = int(os.environ.get("IATMO_ML_MAX", 3))            # jours évalués au plus par passage
ML_VARS = [("heure", [0, 1]), ("week-end", [2]), ("vent", [3]), ("pluie", [4]), ("fond", [5]), ("incrément du modèle", [6]),
           ("trafic (TMJA)", [7]), ("rapport H/L", [8]), ("distance à la bordure", [9])]
ML_MANQ = [3, 4, 7, 8, 9]                                   # variables parfois absentes (perceptron : indicateur « absente »)
# Contraintes de monotonie du gradient boosting sur la correction : croissante avec le trafic, décroissante avec la distance à la
# bordure et, pour le NO₂, avec le vent (dilution ; pour les particules, le vent peut aussi remettre en suspension)
ML_MONO = {"NO2": [0, 0, 0, -1, 0, 0, 0, 1, 0, -1], "PM10": [0, 0, 0, 0, 0, 0, 0, 1, 0, -1], "PM25": [0, 0, 0, 0, 0, 0, 0, 1, 0, -1]}
MLP_GRAINES = (3, 2)                                        # perceptron : graines moyennées pour le test, pour chaque groupe de validation
MLP_LIGNES = 20000                                          # perceptron : lignes horaires tirées au plus pour l'apprentissage (temps de calcul)
def ml_matrice(L):
    X = np.full((len(L), 10), np.nan); y = np.zeros(len(L)); base = np.zeros(len(L)); F = np.zeros(len(L)); T = np.zeros(len(L))
    for i, (p, t, J, h, we, u, P, Fi, dref, dc, Ti) in enumerate(L):
        fi = FICHES.get(t) or {}
        X[i, :7] = [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), 1.0 if we else 0.0,
                    u if u is not None else np.nan, P if P is not None else np.nan, Fi, dc]
        if fi.get("tmja"): X[i, 7] = math.log(fi["tmja"])
        if fi.get("A") is not None: X[i, 8] = fi["A"]
        if fi.get("dbord") is not None: X[i, 9] = fi["dbord"]
        base[i] = Fi + dc; F[i] = Fi; T[i] = Ti; y[i] = Ti - base[i]
    return X, y, base, F, T
class PerceptronMoyen:
    """Perceptron multicouche (32 et 16 neurones, ReLU), variables normalisées, absentes remplacées par la moyenne avec un
    indicateur, cible normalisée, arrêt anticipé ; moyenne de plusieurs graines ; au plus MLP_LIGNES lignes tirées au hasard."""
    def __init__(self, graines): self.graines = graines
    def _prep(self, X, fit=False):
        manq = np.isnan(X[:, ML_MANQ]).astype(float); Xi = np.where(np.isnan(X), self.mu, X)
        if fit: self.mn = Xi.mean(axis=0); self.sd = Xi.std(axis=0); self.sd[self.sd == 0] = 1.0
        return np.hstack([(Xi - self.mn) / self.sd, manq])
    def fit(self, X, y):
        if len(X) > MLP_LIGNES: o = np.random.default_rng(1).choice(len(X), MLP_LIGNES, replace=False); X = X[o]; y = y[o]
        with np.errstate(all="ignore"): mu = np.nanmean(X, axis=0)
        self.mu = np.where(np.isnan(mu), 0.0, mu); Z = self._prep(X, True); self.ym = float(y.mean()); self.ys = float(y.std()) or 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # convergence incomplète au bout de 80 passes : sans gravité ici
            self.m = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=80, batch_size=256, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=8, random_state=g).fit(Z, (y - self.ym) / self.ys) for g in range(self.graines)]
        return self
    def predict(self, X):
        Z = self._prep(X); return self.ym + self.ys * np.mean([m.predict(Z) for m in self.m], axis=0)
class GBMFiltre:
    """Gradient boosting sur les seules variables qui varient dans l'apprentissage : scikit-learn 1.9 refuse une variable entièrement
    absente ou constante (vent et pluie sans archive AROME, par exemple après la remise à zéro de la branche donnees)."""
    def __init__(self, p): self.p = p
    def fit(self, X, y):
        self.k = [j for j in range(X.shape[1]) if len(np.unique(X[~np.isnan(X[:, j]), j])) >= 2]
        self.m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
                                               monotonic_cst=[ML_MONO[self.p][j] for j in self.k], random_state=0).fit(X[:, self.k], y)
        return self
    def predict(self, X): return self.m.predict(X[:, self.k])
CANDIDATS = {"gbm": (lambda p, cv: GBMFiltre(p)),
             "mlp": (lambda p, cv: PerceptronMoyen(MLP_GRAINES[1] if cv else MLP_GRAINES[0]))}
def ml_rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def ml_evaluer(J, C):
    if C["nj_obs"] < HIST_MIN: return None
    tr = lignes(C["jours"], C); te = lignes([J], C)
    res = {"lib": "scikit-learn " + SKL + ", HistGradientBoostingRegressor (contraintes de monotonie)", "jours": C["nj_obs"], "test": {}, "cv": {}, "imp": {},
           "mlp": {"lib": "scikit-learn " + SKL + ", MLPRegressor (32, 16), moyenne de %d graines" % MLP_GRAINES[0], "test": {}, "cv": {}, "imp": {}}}
    zas = lambda t: (ST.get(t) or {}).get("zas") or t
    for p in ("NO2", "PM10", "PM25"):
        La = [r for r in tr if r[0] == p]; Lt = [r for r in te if r[0] == p]
        if len(La) < 2000 or len(Lt) < 50: continue
        Xa, ya, _, _, _ = ml_matrice(La); Xt, yt, bt, Ft, Tt = ml_matrice(Lt)
        if p == "NO2":
            gs = sorted({zas(r[1]) for r in La + Lt}, key=lambda g: hashlib.md5(g.encode()).hexdigest()); pli = {g: i % 5 for i, g in enumerate(gs)}
            ga = np.array([pli[zas(r[1])] for r in La]); gt = np.array([pli[zas(r[1])] for r in Lt])
        for cand, fab in CANDIDATS.items():
            R = res if cand == "gbm" else res["mlp"]
            m = fab(p, False).fit(Xa, ya); hy = np.maximum(0.0, bt + m.predict(Xt))
            R["test"][p] = {"n": len(Lt), "hyb": round(ml_rmse(hy, Tt), 2), "biais": round(float(np.mean(hy - Tt)), 2)}
            if p != "NO2": continue
            # Importance par permutation (jour J) : hausse de l'erreur quand une variable est mélangée
            rg = np.random.default_rng(0); e0 = ml_rmse(hy, Tt); imp = {}
            for nom, cols in ML_VARS:
                d = []
                for _ in range(3):
                    Xp = Xt.copy(); o = rg.permutation(len(Xp)); Xp[:, cols] = Xt[o][:, cols]
                    d.append(ml_rmse(np.maximum(0.0, bt + m.predict(Xp)), Tt) - e0)
                imp[nom] = round(float(np.mean(d)), 2)
            R["imp"][p] = imp
            # Validation croisée par agglomération écartée (5 groupes de ZAS)
            se = [0.0, 0.0, 0.0]; n = 0
            for f in range(5):
                mt = gt == f
                if not mt.any() or (ga != f).sum() < 1000: continue
                mf = fab(p, True).fit(Xa[ga != f], ya[ga != f]); hf = np.maximum(0.0, bt[mt] + mf.predict(Xt[mt]))
                se[0] += float(np.sum((Ft[mt] - Tt[mt]) ** 2)); se[1] += float(np.sum((bt[mt] - Tt[mt]) ** 2)); se[2] += float(np.sum((hf - Tt[mt]) ** 2)); n += int(mt.sum())
            if n: R["cv"][p] = {"n": n, "fond": round(math.sqrt(se[0] / n), 2), "cale": round(math.sqrt(se[1] / n), 2), "hyb": round(math.sqrt(se[2] / n), 2), "groupes": len(gs)}
    return res
fcj = os.path.join(OUT, "suivi/calage_jours.json"); CJ = lire_json(fcj, [])
NFICHES = sum(1 for f in FICHES.values() if f.get("tmja"))
faits_cj = {e["jour"] for e in CJ if e.get("v") == 1 and "p5" in (e.get("p") or {})}   # jours antérieurs à P5 : recalculés (au plus RATTRAPAGE par passage)
CJC = {}
for J in [j for j in complets[-RATTRAPAGE:] if j not in faits_cj]:
    t1 = time.time(); C, IC = caler([j for j in complets if j < J][-30:], boot=True)
    e = {"v": 1, "jour": J, "fen": C["nj_obs"], "paires": len(paires), "fiches": NFICHES, "nst": C["nst"], "nh": C["n"],
         "p": {"D": C["D"], "a": C["fiches"]["a"] if C["fiches"] else None, "c": C["fiches"]["c"] if C["fiches"] else None,
               "Df": C["fiches"]["D"] if C["fiches"] else None, "vent": C["vent"]["a"] if C["vent"] else None,
               "pluie": C["pluie"]["k"] if C["pluie"] else None,
               "p5": {"D": C["D"].get("NOX"), "vent": ((C.get("p5") or {}).get("vent") or {}).get("a"),
                      "temp": ((C.get("p5") or {}).get("temp") or {}).get("g"), "chimie_30j": (C.get("p5") or {}).get("chimie_30j")}},
         "ic": IC, "test": tester(J, C) if C["D"] else {}, "arome": J in AR}
    CJ[:] = [x for x in CJ if x["jour"] != J]; CJ.append(e); CJC[J] = C; print("progression du calage", J, ": fenêtre", e["fen"], "jours ; test", e["test"], ";", round(time.time() - t1, 1), "s")
# Correction apprise : derniers jours sans évaluation (au plus ML_MAX par passage)
if SKL:
    for e in [e for e in CJ if e["jour"] in complets[-ML_MAX:] and ("ml" not in e or (e["ml"] and "mlp" not in e["ml"]))]:
        t1 = time.time(); J = e["jour"]; C = CJC.get(J) or caler([j for j in complets if j < J][-30:])[0]
        e["ml"] = ml_evaluer(J, C) if C["D"] else None
        print("correction apprise", J, ": gradient boosting", {k: v for k, v in (e["ml"] or {}).items() if k in ("test", "cv")},
              "; perceptron", {k: v for k, v in ((e["ml"] or {}).get("mlp") or {}).items() if k in ("test", "cv")}, ";", round(time.time() - t1, 1), "s")
else: print("correction apprise : scikit-learn absent, évaluation sautée")
CJ = sorted(CJ, key=lambda e: e["jour"])[-400:]
ecrire_json(fcj, CJ)
# Règle d'emploi de P5 dans la carte : actif, sauf si, sur les 7 derniers jours testés à l'aveugle avec la météo AROME (vent et température
# lus pour la moitié des heures au moins, comme sur la carte ; 3 jours au moins), son erreur sur le NO2 dépasse de plus de 5 % celle du
# modèle calé directement sur le NO2 ; la carte revient alors au modèle calé et l'indique. Sans météo, P5 perd ses facteurs du vent et de
# la température, qui compensent la conversion chimique : ces jours ne représentent pas la carte et ne comptent pas.
t5 = [e["test"]["NO2"]["p5"] for e in CJ[-7:] if ((e.get("test") or {}).get("NO2") or {}).get("p5")]
q5 = [x for x in t5 if x.get("meteo", 0) >= 0.5]
if q5:
    n5 = sum(x["n"] for x in q5); ep = math.sqrt(sum(x["n"] * x["p5"] ** 2 for x in q5) / n5); ec = math.sqrt(sum(x["n"] * x["cale"] ** 2 for x in q5) / n5)
    regle = {"jours": len(q5), "sans_meteo": len(t5) - len(q5), "p5": round(ep, 2), "cale": round(ec, 2), "actif": len(q5) < 3 or ep <= 1.05 * ec}
else: regle = {"jours": 0, "sans_meteo": len(t5), "p5": None, "cale": None, "actif": True}
CAL.setdefault("p5", {})["regle"] = regle; ecrire_json(os.path.join(OUT, "calage/calage.json"), CAL)
print("P5 :", {k: v for k, v in CAL["p5"].items() if k != "defauts"})
print("durée totale", round(time.time() - T0, 1), "s")

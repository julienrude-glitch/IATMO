# IATMO : prévision horaire jusqu'à minuit aux stations de fond de métropole (note méthodologique, § 8, étape 4).
# Lancé à chaque passage de la tâche de copie (.github/workflows/copie_lcsqa.yml), dans le dossier de la branche « donnees » :
#   python3 prevision.py prevision_modeles.npz
# Entrées (branche donnees) : archive/lcsqa/AAAA-MM-JJ.csv.gz (mesures horaires, heure TU), arome/AAAA-MM-JJ.bin et ta-AAAA-MM-JJ.bin
# (AROME de Météo-France sur la grille de 0,1°, heures de Paris), et le fichier des modèles appris sur l'historique Geod'air 2023-2026
# (histo/analyses/prevision_production.py) : climatologie de chaque station (mois × type de jour × heure de Paris), coefficients de
# persistance de l'écart α(k), gradient boosting « GBd » qui corrige la référence, classifieur des jours où l'ozone dépasse 130 µg/m³.
# Méthode : Ĉ(t+k) = max(0, Clim(t+k) + α(k)·[C(t) − Clim(t)] + correction apprise), t = dernière heure mesurée, k ≤ 24 h.
# Lendemain (expérimental, depuis le 26/11/2026) : même méthode avec le modèle appris jusqu'à 48 h d'échéance (<POL>_gbd48, α à 48 h), heures 0 à 23 du
# lendemain, clé « lendemain » de la sortie.
# Sortie : prevision/derniere.json (lu par la carte) et prevision/archive/AAAA-MM-JJ-HH.json.gz (prévisions telles qu'émises, 60 jours).
# Seul numpy est nécessaire : les arbres sont parcourus ici, sans scikit-learn.
import sys, os, json, gzip, glob, csv, io, warnings, datetime as dt
import numpy as np
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris"); UTC = dt.timezone.utc
POLS = ("NO2", "O3", "PM10", "PM25"); K = 24
H0_CLASSIFIEUR = (6, 16)            # dernière heure mesurée (heure de Paris) pour laquelle le classifieur de l'ozone a été appris


# ---------------------------------------------------------------- calendrier (identique à l'apprentissage)
def paques(a):
    g = a % 19; c = a // 100; h = (c - c // 4 - (8 * c + 13) // 25 + 19 * g + 15) % 30
    i = h - (h // 28) * (1 - (h // 28) * (29 // (h + 1)) * ((21 - g) // 11)); j = (a + a // 4 + i + 2 - c + c // 4) % 7
    l = i - j; m = 3 + (l + 40) // 44; return dt.date(a, m, l + 28 - 31 * (m // 4))

def feries(annees):
    F = set()
    for a in annees:
        p = paques(a); F |= {dt.date(a, 1, 1), dt.date(a, 5, 1), dt.date(a, 5, 8), dt.date(a, 7, 14), dt.date(a, 8, 15), dt.date(a, 11, 1),
                             dt.date(a, 11, 11), dt.date(a, 12, 25), p + dt.timedelta(days=1), p + dt.timedelta(days=39), p + dt.timedelta(days=50)}
    return F

def calendrier(t0, n):
    """Heure de Paris, type de jour (1 = samedi, dimanche ou férié), jour de l'année et clé de climatologie de chaque heure TU depuis t0."""
    loc = [(t0 + dt.timedelta(hours=k)).astimezone(PARIS) for k in range(n)]
    fer = feries(range(loc[0].year, loc[-1].year + 1))
    HL = np.array([t.hour for t in loc], np.int16); MO = np.array([t.month for t in loc], np.int16)
    TY = np.array([1 if (t.weekday() >= 5 or t.date() in fer) else 0 for t in loc], np.int16)
    DOY = np.array([t.timetuple().tm_yday for t in loc], np.float32)
    return loc, HL, TY, DOY, (MO - 1) * 48 + TY * 24 + HL


# ---------------------------------------------------------------- entrées des modèles (les mêmes fonctions servent à l'apprentissage)
def niveau30(A):
    """Moyenne de l'écart à la climatologie sur les 720 dernières heures (24 valeurs au moins)."""
    S, N = A.shape; v = np.nan_to_num(A); n = np.isfinite(A).astype(np.float32)
    cs = np.concatenate([np.zeros((S, 1)), np.cumsum(v, 1)], 1); cn = np.concatenate([np.zeros((S, 1)), np.cumsum(n, 1)], 1)
    idx = np.arange(N); lo = np.maximum(0, idx + 1 - 720); cnt = cn[:, idx + 1] - cn[:, lo]
    return np.where(cnt >= 24, (cs[:, idx + 1] - cs[:, lo]) / np.maximum(cnt, 1), np.nan).astype(np.float32)

def entrees(s, t, k, HL, TY, DOY, CL, A, L30, M):
    """23 entrées de la correction GBd : échéance, calendrier de l'heure visée, climatologie, écarts récents, niveau des 30 jours,
    météo (0 T, 1 vent, 2 pluie, 3 HR, 4 u, 5 v) à l'heure visée et à l'heure de base, température maximale des 24 heures suivantes."""
    N = A.shape[1]; tt = t + k
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        tmax = np.nanmax(np.stack([M[s, 0, np.minimum(t + j, N - 1)] for j in range(1, K + 1)]), 0)
    F = [k, HL[tt], TY[tt], np.sin(2 * np.pi * DOY[tt] / 365.25), np.cos(2 * np.pi * DOY[tt] / 365.25), CL[s, tt], CL[s, t],
         A[s, t], A[s, t - 1], A[s, t - 2], A[s, np.maximum(tt - 24, 0)], L30[s, t],
         M[s, 0, tt], M[s, 1, tt], M[s, 2, tt], M[s, 3, tt], M[s, 0, t], M[s, 1, t], tmax, M[s, 4, tt], M[s, 5, tt], M[s, 4, t], M[s, 5, t]]
    return np.column_stack(F).astype(np.float32)

def entrees_jour(s, t, h0, X, A, CL, L30, M, TY, DOY):
    """Entrées du classifieur journalier de l'ozone : maximum déjà mesuré, écarts à l'heure de base t (heure de Paris h0), niveau des 30 jours,
    climatologie du jour, calendrier, météo du reste de la journée (Tmax, vent, humidité et vent u, v de 12 à 18 h, pluie) ; h0 en dernière entrée."""
    N = A.shape[1]; jt = t - h0
    col = lambda arr, j: arr[s, np.clip(jt + j, 0, N - 1)]
    passe = np.stack([col(X, j) for j in range(h0 + 1)], 1); clj = np.stack([col(CL, j) for j in range(24)], 1)
    reste = lambda v, a, b: np.stack([M[s, v, np.clip(jt + j, 0, N - 1)] for j in range(a, b)], 1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        F = [np.nanmax(passe, 1), X[s, t], A[s, t], A[s, t - 1], L30[s, t], clj.max(1), CL[s, t],
             np.sin(2 * np.pi * DOY[t] / 365.25), np.cos(2 * np.pi * DOY[t] / 365.25), TY[t],
             np.nanmax(reste(0, h0 + 1, 24), 1), np.nanmean(reste(1, 12, 19), 1), np.nanmean(reste(3, 12, 19), 1), np.nanmin(reste(3, h0 + 1, 24), 1),
             np.nanmean(reste(4, 12, 19), 1), np.nanmean(reste(5, 12, 19), 1), np.nansum(reste(2, h0 + 1, 24), 1), M[s, 0, t], np.full(len(s), h0)]
    return np.column_stack(F).astype(np.float32)


# ---------------------------------------------------------------- forêts d'arbres (export de HistGradientBoosting, parcours numpy)
def exporter(modele, nom):
    """Tableaux d'un modèle HistGradientBoosting de scikit-learn (régression, ou classification à deux classes)."""
    f, s, g, d, m, v, o = [], [], [], [], [], [], []; n = 0
    for arbre in modele._predictors:
        nd = arbre[0].nodes; o.append(n); n += len(nd)
        f.append(np.where(nd["is_leaf"] == 1, -1, nd["feature_idx"]).astype(np.int16)); s.append(nd["num_threshold"].astype(np.float64))
        g.append(nd["left"].astype(np.int32)); d.append(nd["right"].astype(np.int32)); m.append(nd["missing_go_to_left"].astype(np.int8)); v.append(nd["value"].astype(np.float64))
    return {nom + "_f": np.concatenate(f), nom + "_s": np.concatenate(s), nom + "_g": np.concatenate(g), nom + "_d": np.concatenate(d),
            nom + "_m": np.concatenate(m), nom + "_v": np.concatenate(v), nom + "_o": np.array(o, np.int32),
            nom + "_b": np.array([float(np.ravel(modele._baseline_prediction)[0])])}

def foret(z, nom, F):
    """Somme brute des arbres (avant sigmoïde pour un classifieur), comme scikit-learn : x ≤ seuil à gauche, valeur absente selon l'arbre."""
    f, s, g, d, m, v, o = (z[nom + x] for x in ("_f", "_s", "_g", "_d", "_m", "_v", "_o"))
    F = np.asarray(F, np.float64); n = len(F); out = np.full(n, float(z[nom + "_b"][0])); lignes = np.arange(n)
    for o0 in o:
        nd = np.zeros(n, np.int64)
        while True:
            fi = f[o0 + nd]; actif = fi >= 0
            if not actif.any(): break
            x = F[lignes, np.maximum(fi, 0)]
            gauche = np.where(np.isnan(x), m[o0 + nd] == 1, x <= s[o0 + nd])
            nd = np.where(actif, np.where(gauche, g[o0 + nd], d[o0 + nd]), nd)
        out += v[o0 + nd]
    return out


# ---------------------------------------------------------------- calcul en exploitation
def lire_archive(debut, t0, N, codes):
    """Mesures horaires de l'archive du LCSQA (heure TU) depuis la date debut : {polluant: matrice station × heure}."""
    ix = {c: i for i, c in enumerate(codes)}; X = {p: np.full((len(codes), N), np.nan, np.float32) for p in POLS}
    for f in sorted(glob.glob("archive/lcsqa/*.csv.gz")):
        if os.path.basename(f)[:10] < debut: continue
        with gzip.open(f, "rt", encoding="utf-8") as fi:
            for r in csv.DictReader(fi, delimiter=";"):
                i = ix.get(r["code"])
                if i is None: continue
                try: h = dt.datetime.strptime(r["heure"], "%Y-%m-%d %H").replace(tzinfo=UTC)
                except ValueError: continue
                k = int((h - t0).total_seconds() // 3600)
                if not 0 <= k < N: continue
                for p in POLS:
                    v = r.get(p, "")
                    if v not in ("", None):
                        try: X[p][i, k] = float(v)
                        except ValueError: pass
    return X

def lire_arome(loc, lon, lat):
    """Météo AROME (grille de 0,1°) aux stations, heure par heure : tableau station × (T, vent, pluie, HR, u, v) × heure."""
    g = json.load(open("arome/meteo.json", encoding="utf-8"))["grille"]; ni, nj, T = g["ni"], g["nj"], g["ni"] * g["nj"]
    jj = ((g["lat0"] - np.asarray(lat)) / g["pas"]).astype(int); ii = ((np.asarray(lon) - g["lon0"]) / g["pas"]).astype(int)
    dedans = (jj >= 0) & (jj < nj) & (ii >= 0) & (ii < ni); cell = np.where(dedans, jj * ni + ii, 0)
    M = np.full((len(lon), 6, len(loc)), np.nan, np.float32); cache = {}
    def tampon(nom, taille):
        if nom not in cache:
            b = None
            if os.path.exists(nom):
                x = np.frombuffer(open(nom, "rb").read(), np.uint8)
                if x.size == taille: b = x
            cache[nom] = b
        return cache[nom]
    for i, t in enumerate(loc):
        j, h = t.strftime("%Y-%m-%d"), t.hour
        w = tampon(f"arome/{j}.bin", 24 * 3 * T); ta = tampon(f"arome/ta-{j}.bin", 24 * 2 * T)
        if w is not None:
            base = h * 3 * T; u = w[base + cell].view(np.int8).astype(np.float32); v = w[base + T + cell].view(np.int8).astype(np.float32); p = w[base + 2 * T + cell].astype(np.float32)
            ok = (u != -128) & (v != -128) & dedans; u, v = np.where(ok, u / 4, np.nan), np.where(ok, v / 4, np.nan)
            M[:, 1, i] = np.hypot(u, v); M[:, 4, i] = u; M[:, 5, i] = v; M[:, 2, i] = np.where((p != 255) & dedans, p / 10, np.nan)
        if ta is not None:
            base = h * 2 * T; tt = ta[base + cell].view(np.int8).astype(np.float32); hr = ta[base + T + cell].astype(np.float32)
            M[:, 0, i] = np.where((tt != -128) & dedans, tt / 2, np.nan); M[:, 3, i] = np.where((hr != 255) & dedans, hr, np.nan)
    return M

def cle_paris(t): return t.astimezone(PARIS).strftime("%Y/%m/%d %H")

def main(fichier):
    z = np.load(fichier, allow_pickle=False)
    maint = dt.datetime.now(PARIS); J0 = maint.date()
    if os.environ.get("IATMO_MAINTENANT"):                                  # essais sur un jour passé (heure de Paris, AAAA-MM-JJTHH:MM)
        maint = dt.datetime.fromisoformat(os.environ["IATMO_MAINTENANT"]).replace(tzinfo=PARIS); J0 = maint.date()
    t0 = dt.datetime.combine(J0 - dt.timedelta(days=32), dt.time(0), PARIS).astimezone(UTC)
    tf = dt.datetime.combine(J0 + dt.timedelta(days=2), dt.time(0), PARIS).astimezone(UTC)
    N = int((tf - t0).total_seconds() // 3600); loc, HL, TY, DOY, KEY = calendrier(t0, N)
    now_i = int((maint.astimezone(UTC) - t0).total_seconds() // 3600)
    fin_i = int((dt.datetime.combine(J0, dt.time(23), PARIS).astimezone(UTC) - t0).total_seconds() // 3600)   # 23 h de Paris
    tous = sorted({str(c) for p in POLS for c in z[p + "_codes"]})
    X = lire_archive((J0 - dt.timedelta(days=33)).isoformat(), t0, N, tous); ix = {c: i for i, c in enumerate(tous)}
    co = {}
    for p in POLS:
        for c, lo, la in zip(z[p + "_codes"], z[p + "_lon"], z[p + "_lat"]): co[str(c)] = (float(lo), float(la))
    Mt = lire_arome(loc, [co[c][0] for c in tous], [co[c][1] for c in tous])
    sortie = {"version": 1, "modele": str(z["version"]), "emis": maint.isoformat(timespec="seconds"), "jour": J0.isoformat(),
              "heures": [], "base": {}, "stations_par_polluant": {}, "stations": {},
              "methode": "Clim(t+k) + α(k)·[C(t) − Clim(t)] + correction GBd ; météo AROME de la copie (grille de 0,1°)",
              "format": "stations : code → valeurs [NO2, O3, PM10, PM2,5] en µg/m³ pour chaque heure de « heures » (heure de Paris), null si non prévue"}
    val, val1 = {}, {}                                                       # jour même ; lendemain (modèle à 48 h, si présent)
    J1 = J0 + dt.timedelta(days=1)
    deb1_i = int((dt.datetime.combine(J1, dt.time(0), PARIS).astimezone(UTC) - t0).total_seconds() // 3600)
    fin1_i = int((dt.datetime.combine(J1, dt.time(23), PARIS).astimezone(UTC) - t0).total_seconds() // 3600)
    for pi, p in enumerate(POLS):
        codes = [str(c) for c in z[p + "_codes"]]; rows = np.array([ix[c] for c in codes])
        Xp = X[p][rows]; M = Mt[rows]; CL = z[p + "_clim"][:, KEY].astype(np.float32); A = Xp - CL; L30 = niveau30(A); ALK = z[p + "_alpha"]
        n_h = np.isfinite(Xp).sum(0); recent = n_h[max(0, now_i - 48):now_i + 1]
        if not len(recent) or recent.max() < 10: continue
        seuil = 0.4 * recent.max(); h0 = max(i for i in range(max(0, now_i - 48), now_i + 1) if n_h[i] >= seuil)
        h0 = int(h0); sortie["base"][p] = cle_paris(t0 + dt.timedelta(hours=h0))
        ts_st = {}
        for s in range(len(codes)):
            fin_mes = [i for i in range(h0 - 3, h0 + 1) if np.isfinite(A[s, i])]
            if fin_mes: ts_st[s] = fin_mes[-1]
        def prevoir(deb, fin, kmax, nom_gbd, alpha, dest):
            S, T, Kk = [], [], []
            for s, ts in ts_st.items():
                for tt in range(deb, fin + 1):
                    if 1 <= tt - ts <= kmax: S.append(s); T.append(ts); Kk.append(tt - ts)
            if not S: return 0
            S, T, Kk = np.array(S), np.array(T), np.array(Kk)
            pred = np.maximum(0, CL[S, T + Kk] + alpha[Kk] * A[S, T] + foret(z, nom_gbd, entrees(S, T, Kk, HL, TY, DOY, CL, A, L30, M)))
            for s, tt, v in zip(S, T + Kk, pred):
                if np.isfinite(v): dest.setdefault(codes[s], {}).setdefault(int(tt), [None] * 4)[pi] = round(float(v), 1)
            return int(len(set(S.tolist())))
        n0 = prevoir(h0 + 1, fin_i, K, p + "_gbd", ALK, val)
        if n0: sortie["stations_par_polluant"][p] = n0
        if p + "_gbd48_f" in z.files:
            n1 = prevoir(deb1_i, fin1_i, 48, p + "_gbd48", z[p + "_alpha48"], val1)
            if n1: sortie.setdefault("lendemain", {"jour": J1.isoformat(), "experimental": True, "stations_par_polluant": {}})["stations_par_polluant"][p] = n1
        if p == "O3" and "o3_130_f" in z.files:
            hl0 = int(HL[h0])
            if H0_CLASSIFIEUR[0] <= hl0 <= H0_CLASSIFIEUR[1]:
                ok = np.array([s for s in range(len(codes)) if np.isfinite(A[s, h0])])
                if len(ok):
                    Fj = entrees_jour(ok, np.full(len(ok), h0), hl0, Xp, A, CL, L30, M, TY, DOY)
                    pr = 1 / (1 + np.exp(-foret(z, "o3_130", Fj)))
                    sortie["o3_130"] = {"base": sortie["base"][p], "stations": int(len(ok)),                 # probabilités de 0,01 au moins
                                        "probabilites": {codes[s]: round(float(q), 2) for s, q in zip(ok, pr) if q >= 0.01}}
    heures = sorted({i for r in val.values() for i in r}); sortie["heures"] = [cle_paris(t0 + dt.timedelta(hours=i)) for i in heures]
    for c, r in val.items(): sortie["stations"][c] = [r.get(i) for i in heures]
    if val1:                                                                 # lendemain, 0 h à 23 h de Paris : même format que le jour même
        h1 = list(range(deb1_i, fin1_i + 1)); L = sortie["lendemain"]
        L["heures"] = [cle_paris(t0 + dt.timedelta(hours=i)) for i in h1]; L["stations"] = {c: [r.get(i) for i in h1] for c, r in val1.items()}
    os.makedirs("prevision/archive", exist_ok=True)
    txt = json.dumps(sortie, ensure_ascii=False, separators=(",", ":"))
    open("prevision/derniere.json", "w", encoding="utf-8").write(txt)
    fa = f"prevision/archive/{J0.isoformat()}-{maint.hour:02d}.json.gz"
    if not os.path.exists(fa):
        with gzip.open(fa, "wt", encoding="utf-8") as g: g.write(txt)
    lim = (J0 - dt.timedelta(days=60)).isoformat()
    for f in glob.glob("prevision/archive/*.json.gz"):
        if os.path.basename(f)[:10] < lim: os.remove(f)
    print("prévision :", {p: (sortie["base"].get(p), sortie["stations_par_polluant"].get(p)) for p in POLS}, "| heures", (sortie["heures"] or ["—"])[0], "→", (sortie["heures"] or ["—"])[-1],
          "| ozone > 130 :", len((sortie.get("o3_130") or {}).get("probabilites", {})), "stations",
          "| lendemain :", (sortie.get("lendemain") or {}).get("stations_par_polluant"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prevision_modeles.npz")

import sitecustomize
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import bisect, csv, hashlib, json, os, random, re, secrets, time, unicodedata
import requests
from datetime import datetime
import joblib, numpy as np, onnxruntime as rt, pandas as pd
import sklearn, stripe
from typing import Any, Dict, List, Tuple, Optional
os.makedirs("/data", exist_ok=True)

import db as chillsdb
import auth as chillsauth
chillsdb.init_db()

# ═══════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════
a = FastAPI()

@a.middleware("http")
async def security_gate(req: Request, call_next):
    path = req.url.path
    if path in ("/legacy-start", "/intake", "/start"):
        return JSONResponse({"error": "not found"}, status_code=404)
    if (path.startswith("/_debug/") or path == "/download-logs") and not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    return await call_next(req)

b = os.path.dirname(__file__)
t = Jinja2Templates(directory=os.path.join(b, "templates"))
# felix css contains "{#" sequences, so jinja comments move to a token that never appears
t.env.comment_start_string = "<%#"
t.env.comment_end_string = "#%>"
if os.path.isdir(os.path.join(b, "images")):
    a.mount("/images", StaticFiles(directory=os.path.join(b, "images")), name="images")
    a.mount("/static", StaticFiles(directory=os.path.join(b, "static"), check_dir=False), name="static")

ff = os.path.join(b, "new_features.json")
pf = os.path.join(b, "new_preprocessor.joblib")
mf = os.path.join(b, "new_model.onnx")
csr = os.path.join(b, "chills_score_reference.json")
mrf = os.path.join(b, "match_reference.json")
qfs = [
    "Minimal Questinonaire.csv",
    "Minimal Questionnaire.csv",
    "minimal_questionnaire.csv",
    "minimal_questionare.csv",
]
sfs = ["Stimuli.csv", "stimuli.csv", "STIMULI.csv"]

E = {"msg": "", "when": "", "sklearn": sklearn.__version__}
P = {
    "onnx_called": False,
    "in_shape": None,
    "out_shape": None,
    "out_names": None,
    "probs": None,
    "argmax": None,
    "last_top5": [],
    "chosen_head_idx": None,     
    "chosen_head_name": None,    
}
LP = "/data/debug_probs.csv"
FW = {"pairs": [], "missing": [], "raw_answers": {}, "built_vector": []}


STIM = [
    "Great Dictator",
    "Think Too Much Feel Too Little (audio)",
    "Perfect Planet",
    "Aramaic (Audio)",
    "Unsung Hero (Thai Insurance)",
    "Interstellar",
    "Dead Poets",
    "Great Dictator (Audio)",
    "Dead Poets (Audio)",
    "Feynman (audio)",
    "Agnus Dei (Audio)",
    "Miserere Me (Audio)",
    "3rd Grade Drop Out (Audio)",
    "Unbroken (Audio)",
    "Laughing Heart (Audio)",
    "Hallelujah Choir (Audio)",
    "Jason Silva (Audio)",
    "Clair de Lune (Audio)",
    "Pale Blue Dot (Audio)",
    "Motorcycle Diaries (Audio)",
    "Pema Chodron (Audio)",
    "Duo Des Fleurs (Audio)",
    "Radiohead Reckoner (Audio)",
    "Sigur Ros - Hoppipolla (Audio)",
    "Wild Geese (Audio)",
    "Air France",
    "Be Kind",
    "Mr. Rogers Testimony",
    "Cloud Atlas",
    "A Thing About Life",
    "Remember the Titans",
    "Amelie",
    "Thai Medicine",
    "Muhammad Ali",
    "Italy Balconies",
    "Mr. Rogers Doc",
    "Hans Zimmer Time",
    "Rocky",
    "Think Too Much Feel Too Little",
    "Aramaic Choir",
]


# ═══════════════════════════════════════════════════
# HELPERS (original)
# ═══════════════════════════════════════════════════
def r1(p):
    try:
        return pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.read_csv(p, encoding="latin-1")

def nm(x): return re.sub(r"[^0-9a-zA-Z_]+", "_", str(x).strip())

def cq(x):
    x = (x or "").replace("\u00e2\u0080\u0099", "'").replace("\u00e2\u0080\u009c", "\u201c").replace("\u00e2\u0080\u009d", "\u201d")
    x = re.sub(r"\s*-\s*\d+\s*$", "", x)
    return x.strip()

def nkey(s):
    if s is None: s = ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("&"," and ")
    s = re.sub(r"[^a-z0-9]+","", s)
    return s

def canon(s):
    if s is None: s = ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("&"," and ").replace("\u2018","'").replace("\u2019","'").replace("\u201c","\"").replace("\u201d","\"")
    s = re.sub(r"\(audio\)","", s)
    s = re.sub(r"[^a-z0-9]+"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s


# ═══════════════════════════════════════════════════
# QUESTIONNAIRE
# ═══════════════════════════════════════════════════
def qcsv():
    p = None
    for z in qfs:
        q = os.path.join(b, z)
        if os.path.exists(q): p = q; break
    if p is None: return []
    d = r1(p)
    d = d.rename(columns={i: i.strip() for i in d.columns})
    h = list(d.columns)
    w = list(d.iloc[0].fillna("").astype(str)) if len(d) >= 1 else h
    u = []
    for i in range(len(h)):
        k = str(h[i]).strip()
        q = str(w[i]).strip() if i < len(w) and w[i] else k
        q = cq(q)
        mn, mx, stp = None, None, "any"
        if k == "Age" or "How old" in k:
            mn, mx, stp = 10, 120, 1; k = "Age"
        elif k.startswith("DPES") or k == "KAMF 4_1":
            mn, mx, stp = 1, 7, 1
        elif k.startswith("NEO-FFI"):
            mn, mx, stp = 1, 5, 1
        u.append({"k": k, "q": q, "n": nm(k), "min": mn, "max": mx, "step": stp, "type": "num"})
    return u

Q = qcsv()

def qdemo():
    return [
        {"k": "Gender", "q": "What is your gender?", "n": "Gender", "type": "sel",
         "opts": ["Male","Female","Other","Prefer not to say"]},
        {"k": "Ethnicity", "q": "What is your ethnicity?", "n": "Ethnicity", "type": "rad",
         "opts": ["Black","White","Hispanic","Asian","Other","Prefer not to say"]},
        {"k": "Education", "q": "What is your highest education level?", "n": "Education", "type": "sel",
         "opts": ["High school","Bachelor","Master","Doctorate","Other","Prefer not to say"]},
        {"k": "Depression", "q": "Have you been diagnosed with depression?", "n": "Depression", "type": "sel",
         "opts": ["No","Yes","Prefer not to say"]},
    ]

NEW_QUESTIONS = [
    {"k": "FREQ_MUSIC_CHILLS", "q": "How often do you get chills or goosebumps from music?", "min": 1, "max": 5},
    {"k": "FREQ_FILM_CHILLS", "q": "How often do you get chills from a film, a speech, or a story?", "min": 1, "max": 5},
    {"k": "FREQ_MOVED_TEARS", "q": "How often are you moved to tears by something beautiful?", "min": 1, "max": 5},
    {"k": "FREQ_MEDITATE", "q": "How often do you meditate?", "min": 1, "max": 5},
    {"k": "FREQ_LUMP_THROAT", "q": "When something moves you, how often do you feel a lump in your throat?", "min": 1, "max": 5},
    {"k": "FREQ_CHEST_WARMTH", "q": "When something moves you, how often do you feel warmth in your chest?", "min": 1, "max": 5},
    {"k": "FREQ_GOOSEBUMPS_MOVED", "q": "How often do you get goosebumps when you are moved?", "min": 1, "max": 5},
    {"k": "DESC_BEAUTY_NOTICE", "q": "I see beauty in things that others might not notice.", "min": 1, "max": 5},
    {"k": "DESC_LOST_THOUGHT", "q": "I like to get lost in thought.", "min": 1, "max": 5},
    {"k": "DESC_VIVID_IMAGINATION", "q": "I have a vivid imagination.", "min": 1, "max": 5},
    {"k": "DESC_DAYDREAM", "q": "I love to daydream.", "min": 1, "max": 5},
    {"k": "DESC_FANTASY", "q": "I enjoy wild flights of fantasy.", "min": 1, "max": 5},
    {"k": "DESC_DEEPER_MEANING_R", "q": "I rarely look for a deeper meaning in things.", "min": 1, "max": 5},
    {"k": "DESC_EMOTIONS_INTENSE", "q": "I experience my emotions intensely.", "min": 1, "max": 5},
    {"k": "DESC_FEEL_OTHERS", "q": "I feel others' emotions.", "min": 1, "max": 5},
    {"k": "DESC_SELDOM_EMOTIONAL_R", "q": "I seldom get emotional.", "min": 1, "max": 5},
    {"k": "DESC_LIKE_MUSIC", "q": "I like music.", "min": 1, "max": 5},
    {"k": "DESC_NATURE_BEAUTY", "q": "I enjoy the beauty of nature.", "min": 1, "max": 5},
    {"k": "DESC_POETRY_R", "q": "I do not like poetry.", "min": 1, "max": 5},
    {"k": "HABIT_MOTIVATIONAL", "q": "How often do you listen to motivational videos or speeches?", "min": 1, "max": 5},
    {"k": "STATE_AROUSAL", "q": "Right now, how calm or excited do you feel?", "min": 1, "max": 5},
    {"k": "STATE_VALENCE", "q": "Right now, how unpleasant or pleasant do you feel?", "min": 1, "max": 5},
]
for _nq in NEW_QUESTIONS:
    _nq.setdefault("n", nm(_nq["k"])); _nq.setdefault("step", 1); _nq.setdefault("type", "num")

def qall():
    u = []; u.extend(Q); u.extend(qdemo()); u.extend(NEW_QUESTIONS); return u


# ═══════════════════════════════════════════════════
# MODEL + PREPROCESSOR LOADING
# ═══════════════════════════════════════════════════
with open(ff, "r", encoding="utf-8") as f:
    F = json.load(f)["features"]

pre = joblib.load(pf)
for c, d in [("_name_to_fitted_passthrough", {}), ("_remainder", "drop")]:
    if not hasattr(pre, c):
        try: setattr(pre, c, d)
        except: pass

sess = rt.InferenceSession(mf, providers=["CPUExecutionProvider"])
inn = sess.get_inputs()[0].name

with open(csr, "r", encoding="utf-8") as f:
    SCORE_REF = sorted(json.load(f)["sorted_scores"])
with open(mrf, "r", encoding="utf-8") as f:
    MATCH_REF = sorted(json.load(f)["sorted_scores"])

# v49 hub: the profile histogram is the reference distribution in 25 bins,
# same geometry as the mockup (x = i*25.04, w 19, bottom 140, max height 118).
def _build_hist():
    n_bins = 25
    lo, hi = SCORE_REF[0], SCORE_REF[-1]
    width = ((hi - lo) / n_bins) or 1.0
    counts = [0] * n_bins
    for v in SCORE_REF:
        counts[min(n_bins - 1, int((v - lo) / width))] += 1
    return lo, width, counts
HIST_LO, HIST_W, HIST_COUNTS = _build_hist()

def hist_bin_of(score: float) -> int:
    return min(24, max(0, int((float(score or 0) - HIST_LO) / HIST_W)))

def hist_bars_for(score: float):
    mx = max(HIST_COUNTS) or 1
    ub = hist_bin_of(score)
    bars = []
    for i, cnt in enumerate(HIST_COUNTS):
        h = round(3 + 115 * cnt / mx, 1)
        bars.append({"x": round(i * 25.04, 1), "y": round(140 - h, 1), "h": h,
                     "fill": "#6a3f37" if i == ub else "rgba(106,63,55,.18)"})
    return bars, round(ub * 25.04 + 9.5, 1)

def one_in_for(score: float) -> int:
    cnt = HIST_COUNTS[hist_bin_of(score)]
    return max(1, round(len(SCORE_REF) / cnt)) if cnt else len(SCORE_REF)

def mean_p_of(user) -> float:
    """users.score holds the TOP-video p; the ChillsScore histogram runs on the
    MEAN p, which lives in vector_json (40 probs). Legacy rows without a valid
    vector fall back to the stored percentile's position in the reference."""
    try:
        v = json.loads(user["vector_json"] or "[]")
        if isinstance(v, list) and len(v) == 40:
            vv = [float(x) for x in v]
            return sum(vv) / len(vv)
    except Exception:
        pass
    pct = float(user["percentile"] or 0.0)
    idx = min(len(SCORE_REF) - 1, max(0, round(pct / 100 * (len(SCORE_REF) - 1))))
    return SCORE_REF[idx]

def req_base(req: Request) -> str:
    """Link base from the live request, so share links work on the onrender
    URL today and on chillstv.com automatically once DNS points at Render."""
    host = req.headers.get("host", "chillstv.com")
    scheme = req.headers.get("x-forwarded-proto", "https")
    return f"{scheme}://{host}"


def log_ev(event: str, user=None, detail: dict = None, pid: str = ""):
    """Every user action lands in the events table. detail is a plain dict."""
    chillsdb.log_event(
        event,
        user_id=user["id"] if user else None,
        pid=(user["pid"] if user else pid) or "",
        detail=json.dumps(detail or {}),
    )

def percentile_against(ref, value):
    """Share of the reference distribution this value beats, 0 to 100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not ref or not np.isfinite(v):
        return 0.0
    return round(100.0 * bisect.bisect_left(ref, v) / len(ref), 1)


# ═══════════════════════════════════════════════════
# STIMULI CSV LOADING
# ═══════════════════════════════════════════════════
def sspath():
    for fn in sfs:
        p = os.path.join(b, fn)
        if os.path.exists(p): return p
    return None

def loadG():
    p = sspath()
    if not p: return pd.DataFrame()
    d = r1(p)
    d = d.rename(columns={str(c).strip(): str(c).strip() for c in d.columns})

    def nn(x): return re.sub(r"[^a-z0-9]+","", str(x).lower())
    m = {nn(c): c for c in d.columns}

    def pick(eq, ct):
        for k in eq:
            if k in m: return m[k]
        for k in ct:
            for z, o in m.items():
                if k in z: return o
        return None

    cu = pick(["url","youtube","link","video_url"], ["url","youtube","link","video"])
    cn = pick(["train_name","name","title","stimulus","original","item","stimulusname"], ["train_name","stimulusname","title","name","stimulus"])
    cd = pick(["description","desc"], ["description","desc"])
    cr = pick(["duration","dur","length"], ["duration","length","dur"])
    cc = pick(["caption","cap"], ["caption","cap"])

    x = pd.DataFrame()
    x["url"]  = d[cu].astype(str) if cu else ""
    x["name"] = d[cn].astype(str) if cn else ""
    x["desc"] = d[cd].astype(str) if cd else ""
    x["dur"]  = d[cr].astype(str) if cr else ""
    x["cap"]  = d[cc].astype(str) if cc else ""

    def fx(u):
        u = (u or "").strip()
        if u and not u.lower().startswith(("http://","https://")): return "https://" + u
        return u

    x["url"] = x["url"].apply(fx)
    x = x.fillna({"url":"","name":"","desc":"","dur":"","cap":""})
    x = x[x["url"].astype(str).str.len() > 0].reset_index(drop=True)
    return x

G = loadG()


# ═══════════════════════════════════════════════════
# STIMULUS MATCHING
# ═══════════════════════════════════════════════════
ALIASES = {
    "mr rogers testimony": ["mr rogers testimony","mr rogers congress testimony","mr rogers senate testimony"],
    "mr rogers doc": ["mr rogers documentary","mr rogers doc"],
    "hans zimmer time": ["time hans zimmer","hans zimmer time"],
    "sigur ros hoppipolla": ["sigur ros hoppipolla","sigur ros hoppipolla audio","sigur ros hoppipolla live"],
    "radiohead reckoner": ["radiohead reckoner"],
    "3rd grade drop out": ["3rd grade drop out","third grade drop out"],
    "think too much feel too little": ["think too much feel too little","think too much feel too little chaplin","great dictator speech think too much"],
    "great dictator": ["the great dictator","great dictator"],
    "dead poets": ["dead poets","dead poets society"],
    "pale blue dot": ["pale blue dot","carl sagan pale blue dot"],
    "feynman": ["feynman","richard feynman","feynman fun to imagine","pleasure of finding things out","fun to imagine","the feynmann series - beauty (audio)"],
    "miserere me": ["miserere me","miserere mei","misere mei, deus (audio)"],
    "thai medicine": ["thai medicine","thai medical ad","thai hospital ad","thai medicine ad"],
}

HARD_MAP = {
    "Feynman (audio)": "The Feynmann Series - Beauty (Audio)",
    "Miserere Me (Audio)": "Misere Mei, Deus (Audio)",
    "Thai Medicine": "Thai Medicine",
}

CSV_NAME = "name"
CSV_URL  = "url"

CSV_IDX: Dict[str,int] = {}
CSV_CANON_ROWS: List[Tuple[int,str]] = []
CSV_BY_SID: Dict[str,int] = {}
for i in range(len(G)):
    c = canon(G.iloc[i].get(CSV_NAME, ""))
    if c:
        CSV_IDX[c] = i
        CSV_CANON_ROWS.append((i, c))
    sid0 = nm(str(G.iloc[i].get(CSV_NAME, "")))
    if sid0:
        CSV_BY_SID[sid0] = i

def video_by_sid(sid: str) -> Optional[Dict]:
    i = CSV_BY_SID.get(sid)
    if i is None:
        return None
    r = G.iloc[i]
    u0 = str(r.get(CSV_URL, "")).strip()
    if u0 and not u0.lower().startswith(("http://", "https://")):
        u0 = "https://" + u0
    return {
        "stimulus_id": sid, "name": str(r.get(CSV_NAME, "")).strip(),
        "url": u0, "desc": str(r.get("desc", "")), "dur": str(r.get("dur", "")),
        "cap": str(r.get("cap", "")),
    }

def hub_video_catalog(limit: int = 12) -> List[Dict]:
    """Real videos for the profile hub grid, drawn from the stimulus catalog."""
    out = []
    for sid, i in CSV_BY_SID.items():
        v = video_by_sid(sid)
        if v and v["url"] and "audio" not in v["name"].lower():
            out.append(v)
        if len(out) >= limit:
            break
    return out

def try_match_name(cn: str) -> Optional[int]:
    for train_name, csv_title in HARD_MAP.items():
        if cn == canon(train_name):
            target = canon(csv_title)
            if target in CSV_IDX:
                return CSV_IDX[target]
    if cn in CSV_IDX:
        return CSV_IDX[cn]
    for k, vs in ALIASES.items():
        if cn == k or cn in vs:
            for v in vs:
                vv = canon(v)
                if vv in CSV_IDX:
                    return CSV_IDX[vv]
    st = set(cn.split())
    best_i, best_s = None, -1.0
    for ri, cc in CSV_CANON_ROWS:
        toks = set(cc.split())
        if not toks: continue
        inter = len(st & toks)
        if inter == 0: continue
        s = inter / max(1, len(st))
        if s > best_s:
            best_s = s; best_i = ri
    if best_i is not None and best_s >= 0.5:
        return best_i
    for ri, cc in CSV_CANON_ROWS:
        if cn in cc or cc in cn:
            return ri
    return None

IDX: Dict[int,int] = {}
UNM = []
for si, sname in enumerate(STIM):
    c = canon(sname)
    got = try_match_name(c)
    if got is None:
        UNM.append({"stim": sname, "canon": c})
    else:
        IDX[si] = got


# ═══════════════════════════════════════════════════
# FEATURE MAPPING + MODEL INFERENCE
# ═══════════════════════════════════════════════════
with open(ff, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)["features"]

AGE_MID = {"18-24": 21.0, "25-34": 30.0, "35-44": 40.0, "45-54": 50.0, "55-64": 60.0, "65+": 70.0,
           "18 to 24": 21.0, "25 to 34": 30.0, "35 to 44": 40.0, "45 to 54": 50.0, "55 to 64": 60.0,
           "65 and over": 70.0}

def age_years_from(v):
    """Bucket -> midpoint used in training. Exact ages clamped to 10..95. Fallback 40."""
    s = str(v or "").strip().lower().replace(" years old", "")
    for k, mid in AGE_MID.items():
        if s.startswith(k):
            return mid
    try:
        return min(max(float(s), 10.0), 95.0)
    except (TypeError, ValueError):
        return 40.0

def sexf_from(v):
    s = str(v or "").strip().lower()
    if s == "female": return 1.0
    if s == "male": return 0.0
    return 0.5

SCALE_ITEMS = [z for z in FEATURES if z not in ("age_years", "sexf") and z.lower() != "stimulus"]

def map_answers_to_features(H):
    """Notebook build_p1: 9 scale items as floats (NaN if unparseable),
    Age -> age_years midpoint, Gender text -> sexf. Order follows FEATURES."""
    m = []; pairs = []; miss = []
    for fk in FEATURES:
        if fk.lower() == "stimulus":
            m.append(0.0); pairs.append([fk, 0.0, "Stimulus"])
        elif fk == "age_years":
            v = age_years_from(H.get("Age", ""))
            m.append(v); pairs.append([fk, v, "Age"])
        elif fk == "sexf":
            v = sexf_from(H.get("Gender", ""))
            m.append(v); pairs.append([fk, v, "Gender"])
        else:
            try:
                v = float(H.get(fk, ""))
            except (TypeError, ValueError):
                v = float("nan"); miss.append(fk)
            m.append(v); pairs.append([fk, v, "item"])
    FW["pairs"] = pairs; FW["missing"] = miss; FW["built_vector"] = m
    return m

def to40X(v):
    z = list(pre.feature_names_in_) if hasattr(pre, "feature_names_in_") else (list(FEATURES) + ["Stimulus"])
    rows = []
    for i in range(len(STIM)):
        h = {}
        for fi, fk in enumerate(FEATURES):
            h[fk] = v[fi] if fi < len(v) else 0
        for stim_key in ("Stimulus","stimulus","item"):
            if stim_key in z: h[stim_key] = STIM[i]
        rows.append(h)
    df = pd.DataFrame(rows)
    for stim_key in ("Stimulus","stimulus","item"):
        if stim_key in z and stim_key not in df.columns:
            df[stim_key] = [STIM[i] for i in range(len(STIM))]
    try:
        X = pre.transform(df.reindex(columns=z, fill_value=0))
    except Exception:
        for k in df.columns:
            try: df[k] = pd.to_numeric(df[k], errors="coerce")
            except: pass
        df = df.fillna(0)
        X = pre.transform(df.reindex(columns=z, fill_value=0))
    if hasattr(X, "toarray"): X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def predict_probs(v):
    """Run the model once: 40 rows (one per stimulus), return p(chills) per video."""
    X = to40X(v)
    y = sess.run(["probabilities"], {inn: X})[0]
    return np.asarray(y)[:, 1].astype(np.float32)

def topk(v, k=1, pid="", p=None):
    if p is None:
        p = predict_probs(v)

    eps = (np.arange(len(STIM)) * 1e-9).astype(np.float32)
    p = p + eps
    if np.max(p) - np.min(p) < 1e-6:
        z = (p - np.mean(p)) / (np.std(p) + 1e-9)
        idx = np.argsort(-z)[:k]
    else:
        idx = np.argsort(-p)[:k]

    try:
        P["onnx_called"] = True
        P["in_shape"] = (len(STIM), 51)
        P["out_shape"] = [(len(STIM), 2)]
        P["out_names"] = ["probabilities"]
        P["probs"] = [float(z) for z in p.tolist()]
        P["argmax"] = int(idx[0]) if len(idx) else None
        P["chosen_head_idx"] = 0
        P["chosen_head_name"] = "probabilities"
        is_new = not os.path.exists(LP)
        with open(LP, "a", encoding="utf-8") as fh:
            if is_new: fh.write("ts,pid,argmax,probs\n")
            fh.write(f"{datetime.utcnow().isoformat()},{pid},{P['argmax']},{P['probs']}\n")
    except Exception:
        pass

    o = []
    for j in idx:
        j = int(j)
        n0 = STIM[j]
        if j in IDX and 0 <= IDX[j] < len(G):
            r = G.iloc[IDX[j]]
            u0 = str(r.get(CSV_URL,"")).strip()
            n1 = str(r.get(CSV_NAME, n0)).strip()
            d0 = str(r.get("desc","")); du = str(r.get("dur","")); c0 = str(r.get("cap",""))
        else:
            u0 = ""; n1 = n0; d0 = ""; du = ""; c0 = ""
        if u0 and not u0.lower().startswith(("http://","https://")):
            u0 = "https://" + u0
        sid = nm(n1)
        o.append({
            "idx": j, "score": float(p[j]), "stimulus_id": sid, "url": u0,
            "name": n1, "stim_name": n0, "desc": d0, "dur": du, "cap": c0
        })
    return o


def _is_video(item: Dict) -> bool:
    name = (item.get("name","") or "").lower()
    url  = (item.get("url","") or "").lower()
    return ("audio" not in name) and (url.startswith("http"))

def _answers_signature(vec: list, places: int = 1) -> str:
    try:
        r = [f"{float(x):.{places}f}" for x in (vec or [])]
    except Exception:
        r = [str(x) for x in (vec or [])]
    return ",".join(r)

def _choose_from_ties(items: List[Dict], pid: str, built_vec: list, tol_abs: float = 1e-3) -> Dict:
    if not items:
        return {}
    best = max(z.get("score", 0.0) for z in items)
    pool = [z for z in items if (best - z.get("score", 0.0)) <= tol_abs]
    vids = [z for z in pool if _is_video(z)]
    pick_from = vids if vids else pool
    sig = (pid or "") + "|" + _answers_signature(built_vec, places=1)
    h = int(hashlib.sha256(sig.encode("utf-8")).hexdigest(), 16)
    return pick_from[h % len(pick_from)]


# ═══════════════════════════════════════════════════
# NEW: STIMULUS PROFILES (from Felix's JSON)
# ═══════════════════════════════════════════════════
_sp_path = os.path.join(b, "stimulus_profiles.json")
if os.path.exists(_sp_path):
    with open(_sp_path, "r", encoding="utf-8") as f:
        SPROF = json.load(f)
else:
    SPROF = {"stimuli": [], "dataset_summary": {}}

# build lookup by canonicalized name
SPROF_IDX: Dict[str, dict] = {}
for _s in SPROF.get("stimuli", []):
    SPROF_IDX[canon(_s.get("name", ""))] = _s


def _parse_scale(s: str) -> Tuple[float, float]:
    s = str(s).replace("\u2013", "-").replace("\u2014", "-")
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0, 100

def _scale_pct(val: float, scale_str: str) -> float:
    lo, hi = _parse_scale(scale_str)
    if hi <= lo: return 50.0
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100.0))

def _to_embed_url(url: str) -> str:
    if not url: return ""
    url = url.strip()
    if "youtube.com/watch?v=" in url:
        return url.replace("watch?v=", "embed/")
    if "youtu.be/" in url:
        vid = url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
        return "https://www.youtube.com/embed/" + vid
    return ""

def build_profile_data(stimulus_name: str, stimulus_url: str = "", stimulus_desc: str = "") -> dict:
    cn = canon(stimulus_name)
    profile = SPROF_IDX.get(cn)
    if not profile:
        for k, v in SPROF_IDX.items():
            if cn in k or k in cn:
                profile = v
                break

    ds = SPROF.get("dataset_summary", {})

    if not profile:
        return {
            "title": stimulus_name,
            "desc": stimulus_desc,
            "url": stimulus_url,
            "embed_url": _to_embed_url(stimulus_url),
            "stats": {"chills_rate_pct": 0, "n_participants": 0, "most_common_age_group": "N/A"},
            "who_matches_paragraph": "",
            "profile_cards": [],
            "dataset_n": "2,937",
            "dataset_url": ds.get("dataset_url", ""),
            "study_url": ds.get("primary_study_url", ""),
        }

    cards = []
    for c in profile.get("profile_cards", []):
        scale_str = c.get("scale", "1-7")
        val = c.get("value", 0)
        avg = c.get("global_avg", 0)
        cards.append({
            **c,
            "pct": round(_scale_pct(val, scale_str), 1),
            "avg_pct": round(_scale_pct(avg, scale_str), 1),
        })

    return {
        "title": profile.get("title", stimulus_name),
        "desc": stimulus_desc,
        "url": stimulus_url,
        "embed_url": _to_embed_url(stimulus_url),
        "stats": profile.get("stats", {}),
        "who_matches_paragraph": profile.get("who_matches_paragraph", ""),
        "profile_cards": cards,
        "dataset_n": "2,937",
        "dataset_url": ds.get("dataset_url", ""),
        "study_url": ds.get("primary_study_url", ""),
    }


# ═══════════════════════════════════════════════════
# NEW: SERVER-SIDE SESSION STORAGE
# ═══════════════════════════════════════════════════
SESSIONS: Dict[str, dict] = {}
SESSION_TTL = 86400  # 24 hours

def _cleanup_sessions():
    now = time.time()
    expired = [k for k, v in SESSIONS.items() if now - v.get("ts", 0) > SESSION_TTL]
    for k in expired:
        del SESSIONS[k]

def _new_session() -> str:
    _cleanup_sessions()
    sid = secrets.token_urlsafe(16)
    SESSIONS[sid] = {"ts": time.time()}
    return sid


# ═══════════════════════════════════════════════════
# NEW: STRIPE SETUP
# ═══════════════════════════════════════════════════
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "price_1T8VZXLFGVq6dtu2SdgbNnAu")


# ═══════════════════════════════════════════════════
# V38 HELPERS - hub, send chills, duo
# ═══════════════════════════════════════════════════

def nav_context(req: Request, user) -> Dict:
    is_admin = chillsauth.is_admin(req)
    if not user:
        avatar_letter = "A" if is_admin else "?"
        return {"avatar_letter": avatar_letter, "notif_count": 0, "is_admin": is_admin, "has_profile": False}
    letter = ((user["pid"] or "").strip()[:1] or "F").upper()
    sends = chillsdb.sends_for_sender(user["id"])
    notif = sum(1 for s in sends if s["status"] == "watched" and not s["sender_seen"])
    return {"avatar_letter": letter, "notif_count": notif, "is_admin": is_admin, "has_profile": True}


def chills_match(v1: List[float], v2: List[float]):
    """Felix's formula: per video, product of the two p(chills); rank by product;
    top 5; raw score = mean of the 5 products. Display value is the percentile of
    the raw score against random study pairs. Returns (raw, percentile, top5_idx)
    or None if either vector is not a valid 40-probability vector (legacy users)."""
    try:
        x = np.asarray(v1, dtype=np.float64)
        y = np.asarray(v2, dtype=np.float64)
    except Exception:
        return None
    if x.shape != (len(STIM),) or y.shape != (len(STIM),):
        return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    if x.min() < 0 or x.max() > 1 or y.min() < 0 or y.max() > 1:
        return None
    prod = x * y
    order = np.argsort(-prod)[:5]
    raw = float(np.mean(prod[order]))
    return raw, percentile_against(MATCH_REF, raw), [int(i) for i in order]


def stim_entry(j: int, score: float) -> Dict:
    """Catalog info for STIM index j, same fields topk emits."""
    n0 = STIM[j]
    if j in IDX and 0 <= IDX[j] < len(G):
        r = G.iloc[IDX[j]]
        u0 = str(r.get(CSV_URL, "")).strip()
        n1 = str(r.get(CSV_NAME, n0)).strip()
        d0 = str(r.get("desc", "")); du = str(r.get("dur", "")); c0 = str(r.get("cap", ""))
    else:
        u0 = ""; n1 = n0; d0 = ""; du = ""; c0 = ""
    if u0 and not u0.lower().startswith(("http://", "https://")):
        u0 = "https://" + u0
    return {"idx": j, "score": float(score), "stimulus_id": nm(n1), "url": u0,
            "name": n1, "stim_name": n0, "desc": d0, "dur": du, "cap": c0}


def bet_outcome(send) -> Optional[str]:
    """Who won a round, or None if the recipient hasn't answered yet.
    Sender-picked (the only mode for new sends, Felix ruling Sep 2026):
    friend gets chills -> sender's point; no chills -> algorithm's point.
    Legacy algo-mode rows invert: the algorithm picked, so chills -> its point."""
    exp = send["experienced"]
    if exp is None:
        return None
    if send["mode"] == "algo":
        return "algorithm" if exp else "you"
    return "you" if exp else "algorithm"


def bets_scoreboard(sends) -> Dict:
    """Cumulative You-vs-Algorithm score across all answered rounds."""
    you = 0; algo = 0
    for s in sends:
        w = bet_outcome(s)
        if w == "you": you += 1
        elif w == "algorithm": algo += 1
    return {"you_points": you, "algo_points": algo, "rounds_played": you + algo}


def cosine_match_pct(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 50.0
    a, b = np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 50.0
    cos = float(np.dot(a, b) / (na * nb))
    return round(max(0.0, min(1.0, (cos + 1) / 2)) * 100, 1)


def avatar_of(name: str) -> str:
    return ((name or "").strip()[:1] or "?").upper()


AVATAR_COLORS = ["#e8448e", "#5b8af5", "#9b59d0", "#1a9c5b", "#f0b429", "#0b0f19"]


def avatar_color(seed: str) -> str:
    h = int(hashlib.sha256((seed or "x").encode()).hexdigest(), 16)
    return AVATAR_COLORS[h % len(AVATAR_COLORS)]


# ═══════════════════════════════════════════════════
# V50 UNIFICATION HELPERS (rewire.bio site + account flow)
# ═══════════════════════════════════════════════════

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
MATCH_BAR = float(os.getenv("MATCH_BAR", "0.65"))
TERMS_VERSION = os.getenv("TERMS_VERSION", "1.1")
PRIVACY_VERSION = os.getenv("PRIVACY_VERSION", "1.0")

FLOW_GRADIENTS = [
    "radial-gradient(120% 90% at 30% 20%,#2a2438 0%,#141320 55%,#0c0b12 100%)",
    "linear-gradient(135deg,#1a1a2e,#0f3460)",
    "linear-gradient(135deg,#1c2a4a,#3a1c4a)",
    "linear-gradient(135deg,#301a2e,#1e2a66)",
    "linear-gradient(135deg,#233048,#3a2c5a)",
    "linear-gradient(135deg,#101c2e,#0c4a5e)",
]

def ytid(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{6,})", url or "")
    return m.group(1) if m else ""

def _user_initial(user) -> str:
    if not user:
        return ""
    src0 = (user["pid"] or "").strip() or (user["email"] or "").strip()
    return (src0[:1] or "?").upper()

def _user_has_profile(user) -> bool:
    return bool(user and (user["stimulus_id"] or ""))

def flow_videos_for(user) -> list:
    """The top 5 as the unified template consumes them: sid, title, desc,
    length, kind, poster gradient, url, youtube id."""
    if not user:
        return []
    try:
        top5 = json.loads(user["top5_json"] or "[]")
    except Exception:
        top5 = []
    out = []
    for i, e in enumerate(top5[:5]):
        sid = e.get("stimulus_id", "")
        cat = video_by_sid(sid) or {}
        url = e.get("url", "") or cat.get("url", "")
        out.append({
            "sid": sid,
            "t": e.get("name", "") or cat.get("name", ""),
            "d": cat.get("desc", ""),
            "len": cat.get("dur", ""),
            "kind": "Picked for you",
            "bg": FLOW_GRADIENTS[i % len(FLOW_GRADIENTS)],
            "url": url,
            "yt": ytid(url),
        })
    return out

def next_unseen_pick(user, exclude_sid: str = ""):
    """Next unseen pick from the user's top 5, for the after-video No path.
    Seen = watched rows + anything they answered about. None when the 5 are done."""
    try:
        top5 = json.loads(user["top5_json"] or "[]")
    except Exception:
        top5 = []
    if not top5:
        return None
    seen = {exclude_sid} if exclude_sid else set()
    with chillsdb.get_conn() as conn:
        for r in conn.execute("SELECT stimulus_id FROM video_watches WHERE user_id=?", (user["id"],)):
            seen.add(r["stimulus_id"])
    for r in chillsdb.after_answers_for_user(user["id"], limit=200):
        seen.add(r["stimulus_id"])
    for rank, e in enumerate(top5[:5]):
        sid = e.get("stimulus_id", "")
        cat = video_by_sid(sid) or {}
        url = e.get("url", "") or cat.get("url", "")
        if not sid or sid in seen or not url.startswith("http"):
            continue
        return {
            "sid": sid, "t": e.get("name", "") or cat.get("name", ""), "d": cat.get("desc", ""),
            "len": cat.get("dur", ""), "kind": "Another pick for you",
            "bg": FLOW_GRADIENTS[rank % len(FLOW_GRADIENTS)],
            "url": url, "yt": ytid(url),
        }
    return None


def _lab_items(user_id: int) -> list:
    """The user's own lab submissions for the profile card, newest first."""
    out = []
    for c in chillsdb.contributions_by_user(user_id):
        out.append({
            "url": c["url"] or "",
            "text": c["description"] or "",
            "date": datetime.utcfromtimestamp(c["created_at"]).strftime("%b %d"),
        })
    return out


def _duo_cv(row, initiator=None, partner=None):
    """Joint top five for a duo pair with per person chills flags. Entries
    carry the same fields as profile videos so the SPA can play them."""
    initiator = initiator or chillsdb.get_user_by_id(row["user_id"])
    partner = partner or (chillsdb.get_user_by_id(row["partner_user_id"]) if row["partner_user_id"] else None)
    a_map = chillsdb.chills_by_video(row["user_id"])
    b_map = chillsdb.chills_by_video(row["partner_user_id"] or 0)
    try:
        va = json.loads(initiator["vector_json"] or "[]") if initiator else []
        vb = json.loads(partner["vector_json"] or "[]") if partner else []
    except Exception:
        va, vb = [], []
    res = chills_match(va, vb)
    pool = []
    if res is not None:
        for rank, jdx in enumerate(res[2]):
            e = stim_entry(jdx, 0.0)
            pool.append({"sid": e.get("stimulus_id", ""), "t": e.get("name", ""),
                         "d": e.get("desc", ""), "len": e.get("dur", ""), "kind": "Picked for you",
                         "bg": FLOW_GRADIENTS[rank % len(FLOW_GRADIENTS)],
                         "url": e.get("url", ""), "yt": ytid(e.get("url", ""))})
    else:
        try:
            top5 = json.loads(initiator["top5_json"] or "[]") if initiator else []
        except Exception:
            top5 = []
        for rank, e in enumerate(top5[:5]):
            v0 = video_by_sid(e.get("stimulus_id", "")) or {}
            url = e.get("url", "") or v0.get("url", "")
            pool.append({"sid": e.get("stimulus_id", ""), "t": e.get("name", "") or v0.get("name", ""),
                         "d": v0.get("desc", ""), "len": v0.get("dur", ""), "kind": "Picked for you",
                         "bg": FLOW_GRADIENTS[rank % len(FLOW_GRADIENTS)], "url": url, "yt": ytid(url)})
    cv = []
    for e in pool[:5]:
        sid0 = e["sid"]
        cv.append({**e, "a": bool(a_map.get(sid0)), "b": bool(b_map.get(sid0)),
                    "both": bool(a_map.get(sid0)) and bool(b_map.get(sid0))})
    return cv


def unified_context(req: Request, user) -> dict:
    """Everything unified.html needs, for both jinja and the CTX json."""
    has_profile = _user_has_profile(user)
    likely = 0
    one_in = 8
    bars, marker_x = [], 0.0
    videos = []
    latest_duo = None
    duo_results, duo_waiting, duo_link = [], [], ""
    you_points, algo_points = 0, 0
    if user and has_profile:
        likely = max(1, min(99, round(float(user["percentile"] or 0))))
        one_in = one_in_for(mean_p_of(user))
        bars, marker_x = hist_bars_for(mean_p_of(user))
        videos = flow_videos_for(user)
        for d0 in chillsdb.duos_involving_user(user["id"]):
            if d0["status"] != "completed":
                continue
            if d0["user_id"] == user["id"]:
                other = d0["partner_name"] or "A friend"
            else:
                initiator0 = chillsdb.get_user_by_id(d0["user_id"])
                other = (initiator0["pid"] if initiator0 else "") or "A friend"
            cv0 = _duo_cv(d0)
            duo_results.append({
                "token": d0["token"], "partner_name": other,
                "partner_letter": avatar_of(other or "?"),
                "match_pct": float(d0["match_pct"] or 0),
                "date_str": datetime.utcfromtimestamp(d0["completed_at"] or d0["created_at"]).strftime("%b %d"),
                "matches": sum(1 for e in cv0 if e["both"]), "cv": cv0,
            })
        if duo_results:
            latest_duo = duo_results[0]
        for d0 in chillsdb.duos_for_user(user["id"]):
            if d0["status"] == "pending":
                duo_waiting.append({"token": d0["token"], "opened": bool(d0["opened_at"]),
                                     "date_str": datetime.utcfromtimestamp(d0["created_at"]).strftime("%b %d")})
        if duo_waiting:
            duo_link = duo_waiting[0]["token"]
        else:
            duo_link = chillsdb.create_duo(user["id"])["token"]
            log_ev("duo_created", user=user, detail={"token": duo_link, "from": "profile"})
            duo_waiting.append({"token": duo_link, "opened": False,
                                 "date_str": datetime.utcnow().strftime("%b %d")})
        board = bets_scoreboard([dict(r) for r in chillsdb.responses_for_sender(user["id"])])
        you_points, algo_points = board["you_points"], board["algo_points"]
    pending_sid = (user["pending_after_sid"] or "") if user else ""
    ctx = {
        "logged_in": chillsauth.is_logged_in(req),
        "has_profile": has_profile,
        "email": (user["email"] or "") if user else "",
        "initial": _user_initial(user),
        "app_access": (user["beta_status"] or "none") if user else "none",
        "edge_code": (user["edge_code"] or "") if user else "",
        "pending_after": bool(pending_sid),
        "pending_sid": pending_sid,
        "likely_pct": likely,
        "videos": videos,
        "lab_items": _lab_items(user["id"]) if user else [],
        "you_points": you_points, "algo_points": algo_points,
        "duo_results": duo_results, "duo_waiting": duo_waiting, "duo_link": duo_link,
    }
    return {
        "request": req, "ctx": ctx, "initial": _user_initial(user) or "?",
        "likely_pct": likely, "one_in": one_in,
        "hist_bars": bars, "marker_x": marker_x, "videos": videos,
        "latest_duo": latest_duo, "you_points": you_points, "algo_points": algo_points,
        "duo_results": duo_results, "duo_waiting": duo_waiting, "duo_link": duo_link,
    }

def _persist_flow(req: Request, user, sess_data):
    """New-funnel version of _finalize_paid_session: writes the questionnaire
    result to the signed-in account's own users row (no row rotation, the
    account is the identity), then resolves a pending send or duo link."""
    stim = sess_data.get("stimulus", {})
    pid = sess_data.get("pid", "")
    if pid and (user["pid"] or "") != pid:
        with chillsdb.get_conn() as _conn:
            _conn.execute("UPDATE users SET pid=? WHERE token=?", (pid, user["token"]))
    top5 = sess_data.get("top5", [])
    top5_entries = [
        {"stimulus_id": z.get("stimulus_id",""), "name": z.get("name",""),
         "url": z.get("url",""), "score": float(z.get("score",0.0))}
        for z in top5
    ]
    official_sid = stim.get("stimulus_id", "")
    top5_entries = [e for e in top5_entries if e["stimulus_id"] != official_sid]
    if official_sid:
        top5_entries.insert(0, {
            "stimulus_id": official_sid, "name": stim.get("stim_name", stim.get("name", "")),
            "url": stim.get("url", ""), "score": float(stim.get("score", 0.0)),
        })
    chillsdb.update_user_match(
        user["token"], stim.get("stimulus_id",""), stim.get("stim_name", stim.get("name","")),
        stim.get("url",""), float(stim.get("score", 0.0)), 0.0, paid=True,
        top5_json=json.dumps(top5_entries),
        vector_json=json.dumps(sess_data.get("probs", [])),
        answers_json=json.dumps(sess_data.get("answers") or {}),
    )
    chillsdb.set_percentile(user["token"], float(sess_data.get("chills_score", 0.0)))
    user = chillsdb.get_user_by_token(user["token"])
    try:
        lp = "/data/logs.csv"
        hdr = ["ts","participant_id","email","prolific_id","stimulus_id","url",
               "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10","description"]
        is_new = not os.path.exists(lp)
        with open(lp, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if is_new: w.writerow(hdr)
            w.writerow([
                datetime.utcnow().isoformat(), pid, user["email"] or "", "",
                stim.get("stimulus_id",""), stim.get("url",""), "unlock", 0, 0, 0, "unified_funnel",
            ])
    except Exception:
        pass
    log_ev("profile_created", user=user, detail={
        "chills_score": float(sess_data.get("chills_score", 0.0)),
        "top_match": stim.get("stim_name", stim.get("name", "")),
        "top5": [e["name"] for e in top5_entries], "funnel": "unified",
    })
    pending_raw = req.cookies.get("pending_link", "")
    if pending_raw:
        try:
            pending = json.loads(pending_raw)
        except Exception:
            pending = {}
        if pending.get("type") == "send" and pending.get("token"):
            send_row = chillsdb.get_send_by_token(pending["token"])
            if send_row:
                chillsdb.set_send_stimulus(pending["token"], stim.get("stimulus_id",""),
                                            stim.get("stim_name", stim.get("name","")), stim.get("url",""))
                chillsdb.set_pending_send_token(user["token"], pending["token"])
        elif pending.get("type") == "duo" and pending.get("token"):
            duo_row = chillsdb.get_duo_by_token(pending["token"])
            if duo_row and duo_row["status"] == "pending" and duo_row["user_id"] == user["id"]:
                duo_row = None
            if duo_row and duo_row["status"] == "pending":
                initiator = chillsdb.get_user_by_id(duo_row["user_id"])
                init_vec = json.loads(initiator["vector_json"] or "[]") if initiator else []
                res = chills_match(init_vec, sess_data.get("probs", []))
                if res is not None:
                    raw, match_pct, order = res
                    joint = stim_entry(order[0], raw)
                    video_sid = joint["stimulus_id"]
                    video_name = joint["stim_name"]
                else:
                    match_pct = cosine_match_pct(init_vec, sess_data.get("vector", []))
                    video_sid = stim.get("stimulus_id","")
                    video_name = stim.get("stim_name", stim.get("name",""))
                result_token = chillsdb.add_duo_result(
                    duo_row["user_id"], user["id"], pid or "A friend", match_pct,
                    video_sid, video_name,
                )
                log_ev("duo_completed", user=user, detail={
                    "token": result_token, "link_token": pending["token"],
                    "match_pct": float(match_pct),
                    "joint_video": video_name, "initiator_id": duo_row["user_id"],
                })


def _resolve_duo_now(req: Request, user, token: str) -> str:
    """A signed in user with a profile opened a duo link. Their vector is
    already stored, so the match is computed right away, no retake."""
    row = chillsdb.get_duo_by_token(token)
    if not row:
        return "/"
    if row["status"] == "completed":
        return f"/duo/{token}"
    if row["user_id"] == user["id"]:
        return "/duo/new"
    chillsdb.mark_duo_opened(token)
    log_ev("duo_link_opened", user=user, detail={"token": token})
    initiator = chillsdb.get_user_by_id(row["user_id"])
    try:
        init_vec = json.loads(initiator["vector_json"] or "[]") if initiator else []
        my_vec = json.loads(user["vector_json"] or "[]")
    except Exception:
        init_vec, my_vec = [], []
    res = chills_match(init_vec, my_vec)
    if res is not None:
        raw, match_pct, order = res
        joint = stim_entry(order[0], raw)
        video_sid = joint["stimulus_id"]
        video_name = joint["stim_name"]
    else:
        match_pct = cosine_match_pct(init_vec, my_vec)
        video_sid = user["stimulus_id"] or ""
        video_name = user["stimulus_name"] or ""
    result_token = chillsdb.add_duo_result(
        row["user_id"], user["id"], (user["pid"] or "").strip() or "A friend", match_pct,
        video_sid, video_name,
    )
    log_ev("duo_completed", user=user, detail={
        "token": result_token, "link_token": token, "match_pct": float(match_pct),
        "joint_video": video_name, "initiator_id": row["user_id"], "instant": True,
    })
    return "/#/match"


# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════

@a.get("/", response_class=HTMLResponse)
def index(req: Request, send: str = "", us: str = ""):
    user = chillsauth.get_current_user(req)
    if send or us:
        # a user who already has a profile never retakes the test for a link.
        # duo resolves instantly from the stored vectors, a bet link opens
        # the recipient page directly.
        if user and _user_has_profile(user):
            dest = _resolve_duo_now(req, user, us) if us else f"/b/{send}"
            resp = RedirectResponse(dest)
            resp.delete_cookie("pending_link")
            return resp
        resp = RedirectResponse("/#/account")
        kind = "send" if send else "duo"
        resp.set_cookie("pending_link", json.dumps({"type": kind, "token": send or us}),
                         max_age=3600, httponly=True, samesite="lax")
        return resp
    uctx = unified_context(req, user)
    if uctx["ctx"]["logged_in"] and uctx["ctx"]["has_profile"]:
        log_ev("hub_viewed", user=user, detail={"likely_pct": uctx["likely_pct"]})
    return t.TemplateResponse("unified.html", uctx)

@a.get("/chillstv")
def chillstv_page():
    return RedirectResponse("/#/chills")

@a.get("/legacy-start", response_class=HTMLResponse)
def legacy_start(req: Request):
    """The pre-unification questionnaire funnel, kept reachable for debugging."""
    q2 = [x for x in Q if x["k"] != "Age"]
    return t.TemplateResponse("index.html", {"request": req, "DEMO": qdemo(), "QS": q2})


# ── account auth ──────────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

@a.post("/auth/signup")
async def auth_signup(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not EMAIL_RE.match(email):
        return JSONResponse({"ok": False, "error": "That email does not look right."})
    if len(password) < 6:
        return JSONResponse({"ok": False, "error": "Password needs at least 6 characters."})
    if chillsdb.get_user_by_email(email):
        return JSONResponse({"ok": False, "error": "That email already has an account. Sign in instead."})
    ph = chillsauth.hash_password(password)
    visitor_token = req.cookies.get(chillsauth.VISITOR_COOKIE, "")
    existing = chillsdb.get_user_by_token(visitor_token)
    if existing and not (existing["email"] or ""):
        user = chillsdb.attach_account(visitor_token, email, ph)
    else:
        user = chillsdb.create_account_user(email, ph)
    resp = JSONResponse({"ok": True, "has_profile": _user_has_profile(user)})
    chillsauth.start_session(resp, req, user["id"])
    log_ev("account_created", user=user, detail={"method": "password"})
    return resp

@a.post("/auth/signin")
async def auth_signin(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    user = chillsdb.get_user_by_email(email)
    if not user or not chillsauth.verify_password(password, user["password_hash"] or ""):
        return JSONResponse({"ok": False, "error": "Wrong email or password."})
    resp = JSONResponse({
        "ok": True,
        "has_profile": _user_has_profile(user),
        "pending_after": bool(user["pending_after_sid"] or ""),
        "pending_sid": user["pending_after_sid"] or "",
    })
    chillsauth.start_session(resp, req, user["id"])
    log_ev("signed_in", user=user, detail={"method": "password"})
    return resp

@a.post("/auth/logout")
async def auth_logout(req: Request):
    resp = JSONResponse({"ok": True})
    chillsauth.end_session(resp, req)
    return resp

@a.get("/auth/google/start")
def google_start(req: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return RedirectResponse("/?gerr=1#/account")
    state = secrets.token_urlsafe(16)
    from urllib.parse import urlencode
    params = urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": req_base(req) + "/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    })
    resp = RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + params)
    resp.set_cookie("g_state", state, max_age=600, httponly=True, samesite="lax")
    return resp

@a.get("/auth/google/callback")
def google_callback(req: Request, code: str = "", state: str = ""):
    if not code or not state or state != req.cookies.get("g_state", ""):
        return RedirectResponse("/?gerr=1#/account")
    try:
        tok = requests.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": req_base(req) + "/auth/google/callback",
            "grant_type": "authorization_code",
        }, timeout=15).json()
        info = requests.get("https://openidconnect.googleapis.com/v1/userinfo",
                             headers={"Authorization": "Bearer " + tok.get("access_token", "")},
                             timeout=15).json()
    except Exception as ex:
        E["msg"] = f"google oauth: {ex}"; E["when"] = datetime.utcnow().isoformat()
        return RedirectResponse("/?gerr=1#/account")
    sub = str(info.get("sub") or "")
    email = (info.get("email") or "").strip().lower()
    if not sub or not email:
        return RedirectResponse("/?gerr=1#/account")
    user = chillsdb.get_user_by_google_sub(sub)
    created = False
    if not user:
        user = chillsdb.get_user_by_email(email)
        if user:
            chillsdb.attach_account(user["token"], email, "", sub)
            user = chillsdb.get_user_by_token(user["token"])
        else:
            visitor_token = req.cookies.get(chillsauth.VISITOR_COOKIE, "")
            existing = chillsdb.get_user_by_token(visitor_token)
            if existing and not (existing["email"] or ""):
                user = chillsdb.attach_account(visitor_token, email, "", sub)
            else:
                user = chillsdb.create_account_user(email, "", sub)
            created = True
    dest = "/#/profile" if _user_has_profile(user) else "/#/consent"
    resp = RedirectResponse(dest)
    resp.delete_cookie("g_state")
    chillsauth.start_session(resp, req, user["id"])
    log_ev("account_created" if created else "signed_in", user=user, detail={"method": "google"})
    return resp


# ── the unified account flow ─────────────────────────────────────
@a.post("/flow/consent")
async def flow_consent(req: Request):
    user = chillsauth.get_session_user(req)
    if not user:
        return JSONResponse({"ok": False, "error": "Please sign in first."})
    try:
        body = await req.json()
    except Exception:
        body = {}
    boxes = body.get("boxes") or {}
    chillsdb.record_consent(user["id"], TERMS_VERSION, PRIVACY_VERSION,
                             json.dumps(boxes) if boxes else "")
    log_ev("consent_given", user=user, detail={
        "terms_version": TERMS_VERSION, "privacy_version": PRIVACY_VERSION, "boxes": boxes,
    })
    return JSONResponse({"ok": True})

@a.post("/flow/submit")
async def flow_submit(req: Request):
    try:
        user = chillsauth.get_session_user(req)
        if not user:
            return JSONResponse({"ok": False, "error": "Please sign in first."})
        try:
            body = await req.json()
        except Exception:
            body = {}
        name = (body.get("name") or "").strip()
        answers = body.get("answers") or {}
        if not name:
            return JSONResponse({"ok": False, "error": "Please tell us your name."})
        H = {}
        for x in qall():
            H[x["k"]] = str(answers.get(x["k"], "") or "")
        bad = []
        for fk in SCALE_ITEMS:
            try:
                float(H.get(fk, ""))
            except (TypeError, ValueError):
                bad.append(fk)
        if bad:
            return JSONResponse({"ok": False, "error": "Answers incomplete: " + ", ".join(bad)})
        FW["raw_answers"] = H
        if body.get("consent"):
            chillsdb.record_consent(user["id"], TERMS_VERSION, PRIVACY_VERSION, "")
        m = map_answers_to_features(H)
        p = predict_probs(m)
        allranked = topk(m, len(STIM), pid=name, p=p)
        mean_p = float(np.mean(p))
        user_chills_score = percentile_against(SCORE_REF, mean_p)
        # felix 10 sep: rotate the match among everything above the bar for this
        # person; keep hallelujah out of the draw when others qualify so it sits
        # second as the no-chills fallback. below the bar: honest top pick.
        pool = [e for e in allranked if float(e.get("score", 0.0)) >= MATCH_BAR]
        non_halle = [e for e in pool if "hallelujah" not in (e.get("name", "") or "").lower()]
        if non_halle:
            best = random.choice(non_halle)
        elif pool:
            best = pool[0]
        elif allranked:
            best = _choose_from_ties(allranked[:5], name, built_vec=FW.get("built_vector", []))
        else:
            best = {"score": 0.0, "stimulus_id": "", "url": "", "name": "", "desc": "", "dur": "", "cap": ""}
        rest = [e for e in allranked if e.get("stimulus_id") != best.get("stimulus_id")]
        best5 = [best] + rest[:4]
        P["last_top5"] = best5[:]
        sess_data = {
            "pid": name, "stimulus": best, "top5": best5,
            "vector": FW.get("built_vector", []),
            "probs": [float(z) for z in p.tolist()],
            "chills_score": user_chills_score,
            "answers": H,
        }
        log_ev("questionnaire_completed", user=user, detail={
            "answers": H, "probs": [round(float(z), 4) for z in p.tolist()],
            "mean_p": round(mean_p, 4), "chills_score": user_chills_score,
            "top5": [{"name": z.get("name", ""), "p": round(float(z.get("score", 0)), 4)} for z in best5],
            "funnel": "unified",
        })
        _persist_flow(req, user, sess_data)
        user = chillsdb.get_user_by_id(user["id"])
        resp = JSONResponse({
            "ok": True,
            "videos": flow_videos_for(user),
            "likely_pct": max(1, min(99, round(float(user["percentile"] or 0)))),
        })
        resp.delete_cookie("pending_link")
        return resp
    except Exception as ex:
        E["msg"] = f"/flow/submit: {ex}"; E["when"] = datetime.utcnow().isoformat()
        return JSONResponse({"ok": False, "error": "Something went wrong on our side."})

@a.post("/flow/pending")
async def flow_pending(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return JSONResponse({"ok": False})
    try:
        body = await req.json()
    except Exception:
        body = {}
    chillsdb.set_pending_after(user["id"], (body.get("sid") or "").strip())
    return JSONResponse({"ok": True})

@a.post("/flow/after")
async def flow_after(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return JSONResponse({"ok": False, "error": "Please sign in first."})
    try:
        body = await req.json()
    except Exception:
        body = {}
    sid = (body.get("sid") or "").strip() or (user["pending_after_sid"] or "")
    chills = bool(body.get("chills"))
    what = (body.get("what") or "").strip()
    why = (body.get("why") or "").strip()
    if sid:
        chillsdb.record_after_answers(user["id"], sid, chills, what, why)
        chillsdb.record_watch(user["id"], sid)
        text = what if not why else (what + ("\n" if what else "") + why)
        author = (user["pid"] or "").strip() or "You"
        chillsdb.add_video_comment(sid, text, chills, author=author, user_id=user["id"])
        try:
            lp = "/data/logs.csv"
            hdr = ["ts","participant_id","email","prolific_id","stimulus_id","url",
                   "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10","description"]
            is_new = not os.path.exists(lp)
            with open(lp, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if is_new: w.writerow(hdr)
                w.writerow([datetime.utcnow().isoformat(), author, user["email"] or "", "", sid, "",
                            "yes" if chills else "no", 0, 0, 0, text])
        except Exception:
            pass
        if user["pending_send_token"] and (user["stimulus_id"] or "") == sid:
            chillsdb.record_send_response(user["pending_send_token"], chills, recipient_name=author)
            chillsdb.set_pending_send_token(user["token"], "")
    chillsdb.clear_pending_after(user["id"])
    log_ev("after_answers", user=user, detail={
        "stimulus_id": sid, "chills": chills, "what": what, "why": why,
    })
    return JSONResponse({"ok": True})

@a.post("/flow/beta")
async def flow_beta(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return JSONResponse({"ok": False, "error": "Please sign in first."})
    if (user["beta_status"] or "none") != "granted":
        chillsdb.set_beta_status(user["id"], "listed")
        chillsdb.set_beta_requested(user["id"])
        log_ev("beta_joined", user=user)
    return JSONResponse({"ok": True})

@a.post("/flow/next-pick")
async def flow_next_pick(req: Request):
    user = chillsauth.get_current_user(req)
    if not user or not _user_has_profile(user):
        return JSONResponse({"ok": False, "error": "Please sign in first."})
    try:
        d = await req.json()
    except Exception:
        d = {}
    pick = next_unseen_pick(user, exclude_sid=(d.get("sid") or ""))
    if not pick:
        log_ev("next_pick_exhausted", user=user)
        return JSONResponse({"ok": True, "done": True})
    log_ev("next_pick_served", user=user, detail={"sid": pick["sid"], "name": pick["t"]})
    return JSONResponse({"ok": True, "done": False, "video": pick})

@a.post("/flow/lab")
async def flow_lab(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return JSONResponse({"ok": False, "error": "Please sign in first."})
    try:
        d = await req.json()
    except Exception:
        d = {}
    text = (d.get("text") or "").strip()[:2000]
    if not text:
        return JSONResponse({"ok": False, "error": "Write or paste something first."})
    m = re.search(r"https?://\S+", text)
    chillsdb.create_contribution(m.group(0) if m else "", text, user["id"])
    log_ev("lab_submitted", user=user, detail={"has_url": bool(m)})
    return JSONResponse({"ok": True, "items": _lab_items(user["id"])})

@a.get("/account/export")
def account_export(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    data = chillsdb.export_user_data(user["id"])
    log_ev("data_exported", user=user)
    headers = {"Content-Disposition": "attachment; filename=rewire_my_data.json"}
    return JSONResponse(data, headers=headers)

@a.post("/account/delete")
async def account_delete(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return JSONResponse({"ok": False, "error": "Not signed in."})
    try:
        body = await req.json()
    except Exception:
        body = {}
    if not body.get("confirm"):
        return JSONResponse({"ok": False, "error": "Confirmation required."})
    log_ev("account_deleted", detail={"user_id": user["id"]})
    chillsdb.delete_user_account(user["id"])
    resp = JSONResponse({"ok": True})
    chillsauth.end_session(resp, req)
    return resp

@a.post("/intake", response_class=HTMLResponse)
async def intake(req: Request):
    try:
        f = await req.form()
        pid = f.get("pid","")
        h = {x["n"]: f.get(x["n"], "") for x in qdemo()}
        h["Age"] = f.get("Age","")
        q2 = [x for x in Q if x["k"] != "Age"]
        return t.TemplateResponse("scales.html", {"request": req, "PID": pid, "QS": q2, "H": h})
    except Exception as ex:
        E["msg"] = str(ex); E["when"] = datetime.utcnow().isoformat()
        return HTMLResponse(f"<pre>Internal error during /intake\n\n{E['msg']}</pre>", status_code=500)

@a.post("/start")
async def start(req: Request):
    try:
        f = await req.form()
        pid = f.get("pid","")
        H = {}
        for x in qall():
            # template names fields x["n"], but read the raw key too as fallback
            H[x["k"]] = f.get(x["n"], "") or f.get(x["k"], "")
        # scales.html hardcodes the KAMF question as name="EasilyMoved"
        if not H.get("KAMF 4_1"):
            H["KAMF 4_1"] = f.get("EasilyMoved", "")
        FW["raw_answers"] = H

        bad = []
        for fk in SCALE_ITEMS:
            try:
                float(H.get(fk, ""))
            except (TypeError, ValueError):
                bad.append(fk)
        if bad:
            return HTMLResponse(
                "<pre>Answers incomplete or invalid: " + ", ".join(bad) +
                "\n\nPlease go back and answer every question.</pre>", status_code=400)

        m = map_answers_to_features(H)

        p = predict_probs(m)
        best5 = topk(m, 5, pid=pid, p=p)
        P["last_top5"] = best5[:]

        mean_p = float(np.mean(p))
        user_chills_score = percentile_against(SCORE_REF, mean_p)

        best = _choose_from_ties(best5, pid, built_vec=FW.get("built_vector", [])) if best5 else {
            "score": 0.0, "stimulus_id": "", "url": "", "name": "", "desc": "", "dur": "", "cap": ""
        }

        # store in session
        sid = _new_session()
        SESSIONS[sid]["pid"] = pid
        SESSIONS[sid]["stimulus"] = best
        SESSIONS[sid]["top5"] = best5
        SESSIONS[sid]["vector"] = FW.get("built_vector", [])
        SESSIONS[sid]["probs"] = [float(z) for z in p.tolist()]
        SESSIONS[sid]["chills_score"] = user_chills_score
        SESSIONS[sid]["answers"] = H
        SESSIONS[sid]["paid"] = False

        log_ev("questionnaire_completed", pid=pid, detail={
            "answers": H, "probs": [round(float(z), 4) for z in p.tolist()],
            "mean_p": round(mean_p, 4), "chills_score": user_chills_score,
            "top5": [{"name": z.get("name", ""), "p": round(float(z.get("score", 0)), 4)} for z in best5],
        })

        if stripe.api_key:
            return RedirectResponse(f"/paywall?sid={sid}", status_code=303)
        # No payment step for this version (Felix, 2026-08-14): go straight to
        # the hub. Automatically reverts to the paywall the moment a real
        # STRIPE_SECRET_KEY is configured.
        return _finalize_paid_session(req, sid, session_label="no_payment")
    except Exception as ex:
        E["msg"] = str(ex); E["when"] = datetime.utcnow().isoformat()
        return HTMLResponse(f"<pre>Internal error during /start\n\n{E['msg']}\n\nCheck /_debug/feature_wire and /_debug/stim_match</pre>", status_code=500)

@a.get("/research", response_class=HTMLResponse)
def research(req: Request):
    return t.TemplateResponse("research.html", {"request": req})

@a.get("/paywall", response_class=HTMLResponse)
def paywall(req: Request, sid: str = ""):
    if not sid or sid not in SESSIONS:
        return RedirectResponse("/")
    return t.TemplateResponse("paywall.html", {
        "request": req, "session_id": sid, "dev_mode": not bool(stripe.api_key),
    })

@a.post("/create-checkout-session")
async def create_checkout_session(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    sid = body.get("session_id", "")
    if sid not in SESSIONS:
        return JSONResponse({"error": "Invalid session"}, status_code=400)

    host = req.headers.get("host", "chillstv.com")
    scheme = req.headers.get("x-forwarded-proto", "https")
    base = f"{scheme}://{host}"

    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="payment",
            success_url=base + "/payment-complete?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base + f"/paywall?sid={sid}",
            metadata={
                "session_id": sid,
                "product": "chillstv_unlock",
                "user_id": SESSIONS[sid].get("pid", ""),
            },
        )
        return JSONResponse({"url": checkout.url})
    except Exception as ex:
        E["msg"] = str(ex); E["when"] = datetime.utcnow().isoformat()
        return JSONResponse({"error": str(ex)}, status_code=500)

@a.get("/payment-complete", response_class=HTMLResponse)
async def payment_complete(req: Request, session_id: str = ""):
    if not session_id:
        return RedirectResponse("/")

    try:
        stripe_sess = stripe.checkout.Session.retrieve(session_id)
    except Exception:
        return RedirectResponse("/")

    if stripe_sess.payment_status != "paid":
        return RedirectResponse("/")

    sid = (stripe_sess.metadata or {}).get("session_id", "")
    log_ev("payment_completed", pid=SESSIONS.get(sid, {}).get("pid", ""),
           detail={"stripe_session": session_id})
    return _finalize_paid_session(req, sid, session_label=session_id)


@a.get("/_dev/skip-payment", response_class=HTMLResponse)
async def dev_skip_payment(req: Request, sid: str = ""):
    """LOCAL-ONLY shortcut so the v38 experience can be clicked through
    without live Stripe keys. Disabled the moment STRIPE_SECRET_KEY is set,
    so it can never bypass payment in a real deployment."""
    if stripe.api_key:
        return HTMLResponse("Disabled: Stripe is configured on this deployment.", status_code=403)
    return _finalize_paid_session(req, sid, session_label="dev_skip")


def _finalize_paid_session(req: Request, sid: str, session_label: str = ""):
    if sid not in SESSIONS:
        return RedirectResponse("/")

    sess_data = SESSIONS[sid]
    sess_data["paid"] = True

    stim = sess_data.get("stimulus", {})
    pid = sess_data.get("pid", "")
    S = build_profile_data(stim.get("stim_name", stim.get("name", "")), stim.get("url", ""), stim.get("desc", ""))
    S["stimulus_id"] = stim.get("stimulus_id", "")
    S["model_score_pct"] = round(float(stim.get("score", 0.0)) * 100, 1)

    # log the unlock
    try:
        lp = "/data/logs.csv"
        hdr = ["ts","participant_id","email","prolific_id","stimulus_id","url",
               "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10","description"]
        is_new = not os.path.exists(lp)
        with open(lp, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if is_new: w.writerow(hdr)
            w.writerow([
                datetime.utcnow().isoformat(), pid, "", "", stim.get("stimulus_id",""),
                stim.get("url",""), "unlock", 0, 0, 0, f"stripe_session={session_label}"
            ])
    except Exception:
        pass

    # persist a hub profile for this visitor (cookie-based, no signup required).
    # the cookie row is reused only when the submitted name matches the name
    # stored on it. a different name on the same browser is a different person,
    # so they get a fresh row and a rotated cookie instead of inheriting the
    # previous person's results (the old-results and score-overwrite bugs).
    visitor_token = req.cookies.get(chillsauth.VISITOR_COOKIE, "")
    user = chillsdb.get_user_by_token(visitor_token)
    if user:
        stored_pid = (user["pid"] or "").strip().lower()
        submitted_pid = (pid or "").strip().lower()
        if stored_pid and submitted_pid and stored_pid != submitted_pid:
            user = None
    if user:
        if pid and (user["pid"] or "") != pid:
            with chillsdb.get_conn() as _conn:
                _conn.execute("UPDATE users SET pid=? WHERE token=?", (pid, user["token"]))
            user = chillsdb.get_user_by_token(visitor_token)
    else:
        user = chillsdb.create_user(pid=pid, session_id=session_label)
        visitor_token = user["token"]

    # `stim` (the tie-broken official match, e.g. preferring video over audio on
    # a near-tied score) must be the one shown first as "Your top match" on the
    # hub, otherwise clicking that badged video never matches what Send Chills
    # ("trust the algo") and Duo are waiting to close the loop on.
    top5 = sess_data.get("top5", [])
    top5_entries = [
        {"stimulus_id": z.get("stimulus_id",""), "name": z.get("name",""),
         "url": z.get("url",""), "score": float(z.get("score",0.0))}
        for z in top5
    ]
    official_sid = stim.get("stimulus_id", "")
    top5_entries = [e for e in top5_entries if e["stimulus_id"] != official_sid]
    if official_sid:
        top5_entries.insert(0, {
            "stimulus_id": official_sid, "name": stim.get("stim_name", stim.get("name", "")),
            "url": stim.get("url", ""), "score": float(stim.get("score", 0.0)),
        })
    top5_json = json.dumps(top5_entries)
    # vector_json now stores the 40 p(chills), one per stimulus. Duo matching
    # consumes it; legacy rows still hold the old feature vector and fall back.
    vector_json = json.dumps(sess_data.get("probs", []))
    score01 = float(stim.get("score", 0.0))
    chillsdb.update_user_match(
        visitor_token, stim.get("stimulus_id",""), stim.get("stim_name", stim.get("name","")),
        stim.get("url",""), score01, 0.0, paid=True, top5_json=top5_json, vector_json=vector_json,
        answers_json=json.dumps(sess_data.get("answers") or {}),
    )
    # users.percentile now holds the ChillsScore: percentile of the user's mean
    # p(chills) against the 2,937 study participants. Higher is better.
    chillsdb.set_percentile(visitor_token, float(sess_data.get("chills_score", 0.0)))
    log_ev("profile_created", user=chillsdb.get_user_by_token(visitor_token), detail={
        "chills_score": float(sess_data.get("chills_score", 0.0)),
        "top_match": stim.get("stim_name", stim.get("name", "")),
        "top5": [e["name"] for e in top5_entries],
    })

    # resolve a pending Send-Chills ("trust the algo") or Duo-compatibility link, if any
    dest = "/hub"
    pending_raw = req.cookies.get("pending_link", "")
    if pending_raw:
        try:
            pending = json.loads(pending_raw)
        except Exception:
            pending = {}
        if pending.get("type") == "send" and pending.get("token"):
            send_row = chillsdb.get_send_by_token(pending["token"])
            if send_row:
                chillsdb.set_send_stimulus(pending["token"], stim.get("stimulus_id",""),
                                            stim.get("stim_name", stim.get("name","")), stim.get("url",""))
                chillsdb.set_pending_send_token(visitor_token, pending["token"])
        elif pending.get("type") == "duo" and pending.get("token"):
            duo_row = chillsdb.get_duo_by_token(pending["token"])
            if duo_row and duo_row["status"] == "pending" and user and duo_row["user_id"] == user["id"]:
                # the link's creator opened their own duo link on the same
                # browser. pairing a person with themselves is meaningless,
                # so leave the duo pending for the real friend.
                duo_row = None
            if duo_row and duo_row["status"] == "pending":
                initiator = chillsdb.get_user_by_id(duo_row["user_id"])
                init_vec = json.loads(initiator["vector_json"] or "[]") if initiator else []
                res = chills_match(init_vec, sess_data.get("probs", []))
                if res is not None:
                    raw, match_pct, order = res
                    joint = stim_entry(order[0], raw)
                    video_sid = joint["stimulus_id"]
                    video_name = joint["stim_name"]
                else:
                    # legacy initiator row: old-format vector, keep old behavior
                    match_pct = cosine_match_pct(init_vec, sess_data.get("vector", []))
                    video_sid = stim.get("stimulus_id","")
                    video_name = stim.get("stim_name", stim.get("name",""))
                result_token = chillsdb.add_duo_result(
                    duo_row["user_id"], user["id"] if user else 0, pid or "A friend", match_pct,
                    video_sid, video_name,
                )
                log_ev("duo_completed", user=user, detail={
                    "token": result_token, "link_token": pending["token"],
                    "match_pct": float(match_pct),
                    "joint_video": video_name, "initiator_id": duo_row["user_id"],
                })
                dest = f"/duo/{result_token}"

    resp = RedirectResponse(dest, status_code=303)
    resp.set_cookie(chillsauth.VISITOR_COOKIE, visitor_token, max_age=60*60*24*365,
                     httponly=True, samesite="lax")
    resp.delete_cookie("pending_link")
    return resp

@a.post("/chills-response")
async def chills_response(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    sid = body.get("session_id", "")
    experienced = body.get("experienced", "")

    if sid in SESSIONS:
        stim = SESSIONS[sid].get("stimulus", {})
        pid = SESSIONS[sid].get("pid", "")
        try:
            lp = "/data/logs.csv"
            hdr = ["ts","participant_id","email","prolific_id","stimulus_id","url",
                   "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10","description"]
            is_new = not os.path.exists(lp)
            with open(lp, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if is_new: w.writerow(hdr)
                w.writerow([
                    datetime.utcnow().isoformat(), pid, "", "", stim.get("stimulus_id",""),
                    stim.get("url",""), experienced, 0, 0, 0, ""
                ])
        except Exception:
            pass

    return JSONResponse({"status": "ok"})

@a.post("/webhook")
async def stripe_webhook(req: Request):
    payload = await req.body()
    sig = req.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return JSONResponse({"error": "Invalid signature"}, status_code=400)

    if event["type"] == "checkout.session.completed":
        obj = event["data"]["object"]
        sid = (obj.get("metadata") or {}).get("session_id", "")
        if sid in SESSIONS:
            SESSIONS[sid]["paid"] = True

    return JSONResponse({"status": "ok"})


# ═══════════════════════════════════════════════════
# V38 - PROFILE HUB
# ═══════════════════════════════════════════════════

@a.get("/me/{token}", response_class=HTMLResponse)
def resume_profile(req: Request, token: str):
    # permanent personal link. opening it points this browser at that
    # profile again, any device, same bearer token pattern as send and
    # duo links. bridges the one cookie per browser limit until login.
    user = chillsdb.get_user_by_token(token)
    if not user:
        return RedirectResponse("/")
    log_ev("profile_resumed", user=user)
    resp = RedirectResponse("/hub", status_code=303)
    resp.set_cookie(chillsauth.VISITOR_COOKIE, token, max_age=60*60*24*365,
                     httponly=True, samesite="lax")
    return resp


@a.get("/hub")
def hub_redirect(req: Request):
    """The v49 hub is replaced by the unified profile. Games pages still
    link here, so redirect instead of removing. No profile means no profile
    page, so guests land on the home page instead."""
    user = chillsauth.get_current_user(req)
    if not user or not _user_has_profile(user):
        return RedirectResponse("/")
    return RedirectResponse("/#/profile")

@a.get("/hub-v49", response_class=HTMLResponse)
def hub(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    top5 = json.loads(user["top5_json"] or "[]")
    videos = [v for v in top5 if v.get("url")] or hub_video_catalog(6)
    for i, v in enumerate(videos):
        v["sid"] = v.get("stimulus_id") or nm(v.get("name", f"video{i}"))
    sends = chillsdb.sends_for_sender(user["id"])
    sent_count = len(sends)
    hit_count = sum(1 for s in sends if s["experienced"])
    hit_rate = round(100 * hit_count / sent_count) if sent_count else 0
    recent = []
    for s in sends[:3]:
        recent.append({**dict(s), "avatar_letter": avatar_of(s["recipient_name"] or "?"),
                        "avatar_color": avatar_color(s["recipient_name"] or str(s["id"]))})
    # v49 hub context: score card, histogram, latest duo, bets scoreboard
    likely_pct = max(1, min(99, round(float(user["percentile"] or 0))))
    bars, marker_x = hist_bars_for(mean_p_of(user))
    latest_duo = None
    for d0 in chillsdb.duos_for_user(user["id"]):
        if d0["status"] == "completed":
            latest_duo = {"token": d0["token"], "partner_name": (d0["partner_name"] or "A friend"),
                          "partner_letter": avatar_of(d0["partner_name"] or "?"),
                          "match_pct": float(d0["match_pct"] or 0)}
            break
    board = bets_scoreboard([dict(r) for r in chillsdb.responses_for_sender(user["id"])])
    log_ev("hub_viewed", user=user, detail={"likely_pct": likely_pct})
    return t.TemplateResponse("profile.html", {
        "request": req, "page": "hub", **nav_context(req, user), "videos": videos,
        "recent_sends": recent, "sent_count": sent_count, "hit_count": hit_count,
        "hit_rate": hit_rate,
        "likely_pct": likely_pct, "one_in": one_in_for(mean_p_of(user)),
        "hist_bars": bars, "marker_x": marker_x, "latest_duo": latest_duo,
        "you_points": board["you_points"], "algo_points": board["algo_points"],
        "video_urls": [v.get("url", "") for v in videos], "base_url": req_base(req),
    })


# ═══════════════════════════════════════════════════
# V38 - VIDEO PAGE
# ═══════════════════════════════════════════════════

@a.get("/video/{sid}", response_class=HTMLResponse)
def video_page(req: Request, sid: str):
    user = chillsauth.get_current_user(req)
    v = video_by_sid(sid)
    if not v:
        return RedirectResponse("/hub")
    comments = [dict(c, avatar_color=avatar_color(c["author"])) for c in chillsdb.comments_for(sid)]
    log_ev("video_viewed", user=user, detail={"stimulus_id": sid, "name": v["name"]})
    return t.TemplateResponse("video.html", {
        "request": req, "page": "video", **nav_context(req, user),
        "video": v, "embed_url": _to_embed_url(v["url"]), "comments": comments,
        "pid": (user["pid"] if user else "") or "Anonymous",
        "video_urls": [v.get("url", "")], "base_url": req_base(req),
    })


@a.post("/video/{sid}/watched")
def video_watched(req: Request, sid: str):
    """Called when the player fires ENDED or the person taps I watched it.
    Feeds admin watched counts and the duo watch chips."""
    user = chillsauth.get_current_user(req)
    if user and video_by_sid(sid):
        chillsdb.record_watch(user["id"], sid)
    log_ev("video_watched", user=user, detail={"stimulus_id": sid})
    return JSONResponse({"ok": True})


@a.post("/video/{sid}/report")
async def video_report(req: Request, sid: str):
    f = await req.form()
    experienced = (f.get("experienced") or "") == "yes"
    text = (f.get("text") or "").strip()
    user = chillsauth.get_current_user(req)
    author = (user["pid"] if user and user["pid"] else "Anonymous") or "Anonymous"
    if user:
        chillsdb.record_watch(user["id"], sid)
    if text:
        chillsdb.add_video_comment(sid, text, experienced, author=author,
                                    user_id=user["id"] if user else None)
    log_ev("chills_report", user=user, pid=author, detail={
        "stimulus_id": sid, "experienced": experienced, "text": text,
    })

    try:
        lp = "/data/logs.csv"
        hdr = ["ts","participant_id","email","prolific_id","stimulus_id","url",
               "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10","description"]
        is_new = not os.path.exists(lp)
        with open(lp, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if is_new: w.writerow(hdr)
            w.writerow([datetime.utcnow().isoformat(), author, "", "", sid, "",
                        "yes" if experienced else "no", 0, 0, 0, text])
    except Exception:
        pass

    if user and user["pending_send_token"] and user["stimulus_id"] == sid:
        chillsdb.record_send_response(user["pending_send_token"], experienced,
                                       recipient_name=author)
        chillsdb.set_pending_send_token(user["token"], "")

    return RedirectResponse(f"/video/{sid}", status_code=303)


# ═══════════════════════════════════════════════════
# V38 - SEND CHILLS
# ═══════════════════════════════════════════════════

@a.get("/bets", response_class=HTMLResponse)
def bets(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    sends = [dict(s) for s in chillsdb.sends_for_sender(user["id"])]
    responses = [dict(r) for r in chillsdb.responses_for_sender(user["id"])]
    resp_counts = chillsdb.response_counts_map(user["id"])
    unrev = chillsdb.unrevealed_counts_map(user["id"])
    # every response is its own round: chills -> your point, none -> the algorithm's
    for r in responses:
        r["outcome"] = bet_outcome(r)
        r["avatar_letter"] = avatar_of(r["respondent_name"] or "?")
    rounds = [r for r in responses if r["revealed_at"]]
    ready = [s for s in sends if unrev.get(s["token"], 0)]
    for s in ready:
        s["unrevealed"] = unrev.get(s["token"], 0)
    unopened = [s for s in sends if resp_counts.get(s["token"], 0) == 0]
    hit_n = sum(1 for r in rounds if r["experienced"])
    hit_rate = round(100 * hit_n / len(rounds)) if rounds else 0
    board = bets_scoreboard(responses)
    now_ts = time.time()
    for x in unopened:
        days = int((now_ts - (x["created_at"] or now_ts)) // 86400)
        x["sent_ago"] = "today" if days < 1 else ("1 day ago" if days == 1 else f"{days} days ago")
    rp = len(rounds)
    rd = sum(unrev.values())
    if rp == 0 and rd == 0:
        lead_line = "No rounds played yet."
    else:
        played = f"{rp} round played" if rp == 1 else f"{rp} rounds played"
        lead_line = played + (f", {rd} more ready to reveal." if rd else ".")
    return t.TemplateResponse("bets.html", {
        "request": req, "page": "bets", **nav_context(req, user),
        "ready": ready, "unopened": unopened, "rounds": rounds, "hit_rate": hit_rate,
        **board, "ready_count": len(ready), "lead_line": lead_line,
        "resp_counts": resp_counts,
        "video_urls": [x.get("stimulus_url", "") for x in ready] + [x.get("stimulus_url", "") for x in unopened],
    })


@a.get("/send", response_class=HTMLResponse)
def send_new(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    top5 = json.loads(user["top5_json"] or "[]")
    videos = [v for v in top5 if v.get("url")] or hub_video_catalog(6)
    for i, v in enumerate(videos):
        v["sid"] = v.get("stimulus_id") or nm(v.get("name", f"video{i}"))
    return t.TemplateResponse("send_new.html", {
        "request": req, "page": "bet-intro", **nav_context(req, user), "videos": videos,
        "video_urls": [v.get("url", "") for v in videos], "send": None,
    })


@a.post("/send/create")
async def send_create(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    f = await req.form()
    # algo-pick removed (Felix, Sep 2026): every send is sender-picked. The
    # algorithm is only the house you bet against. Legacy algo rows stay readable.
    sid = f.get("stimulus_id", "")
    v = video_by_sid(sid)
    if not v:
        return RedirectResponse("/send", status_code=303)
    row = chillsdb.create_send(user["id"], sid, v.get("name", ""), v.get("url", ""), mode="picked")
    log_ev("send_created", user=user, detail={"token": row["token"], "stimulus_id": sid, "name": v.get("name", "")})
    return RedirectResponse(f"/send/{row['token']}", status_code=303)


@a.get("/send/{token}", response_class=HTMLResponse)
def send_show(req: Request, token: str):
    user = chillsauth.get_current_user(req)
    row = chillsdb.get_send_by_token(token)
    if not user or not row or row["sender_user_id"] != user["id"]:
        return RedirectResponse("/bets")
    top5 = json.loads(user["top5_json"] or "[]")
    videos = [v for v in top5 if v.get("url")] or hub_video_catalog(6)
    for i, v in enumerate(videos):
        v["sid"] = v.get("stimulus_id") or nm(v.get("name", f"video{i}"))
    return t.TemplateResponse("send_new.html", {
        "request": req, "page": "bet-intro", **nav_context(req, user), "send": row,
        "videos": videos, "video_urls": [v.get("url", "") for v in videos],
    })


@a.get("/b/{token}", response_class=HTMLResponse)
def bet_view(req: Request, token: str):
    row = chillsdb.get_send_by_token(token)
    if not row:
        return RedirectResponse("/")
    sender = chillsdb.get_user_by_id(row["sender_user_id"])
    sender_name = (sender["pid"] if sender and sender["pid"] else "Someone") or "Someone"
    recipient = chillsauth.get_current_user(req)
    log_ev("bet_link_opened", user=recipient, detail={"token": token, "sender": sender_name,
                                                       "stimulus": row["stimulus_name"]})
    responses = [dict(r) for r in chillsdb.responses_for_send(token)]
    answered = bool(req.cookies.get("br_" + token, "")) or (
        recipient is not None and any(r["respondent_user_id"] == recipient["id"] for r in responses))
    return t.TemplateResponse("bet_view.html", {
        "request": req, "page": "bet-view", **nav_context(req, recipient),
        "send": row, "sender_name": sender_name,
        "recipient_pid": (recipient["pid"] if recipient else "") or "Anonymous",
        "embed_url": _to_embed_url(row["stimulus_url"]) if row["stimulus_url"] else "",
        "guest": True, "video_urls": [row["stimulus_url"] or ""],
        "responses": responses, "answered": answered,
    })


@a.post("/b/{token}/respond")
async def bet_respond(req: Request, token: str):
    f = await req.form()
    experienced = (f.get("experienced") or "") == "yes"
    row = chillsdb.get_send_by_token(token)
    if not row:
        return RedirectResponse("/", status_code=303)
    viewer = chillsauth.get_current_user(req)
    rname = (viewer["pid"] if viewer and viewer["pid"] else "") or ""
    chillsdb.create_send_response(token, experienced, respondent_name=rname,
                                   respondent_user_id=viewer["id"] if viewer else None)
    if row["status"] == "sent":
        chillsdb.record_send_response(token, experienced, recipient_name=rname)
    log_ev("bet_answered", user=viewer, detail={"token": token, "experienced": experienced})
    resp = RedirectResponse(f"/b/{token}", status_code=303)
    resp.set_cookie("br_" + token, "1", max_age=60*60*24*365, httponly=True, samesite="lax")
    return resp


@a.get("/reveal/{token}", response_class=HTMLResponse)
def reveal_gate(req: Request, token: str):
    user = chillsauth.get_current_user(req)
    row = chillsdb.get_send_by_token(token)
    if not user or not row or row["sender_user_id"] != user["id"]:
        return RedirectResponse("/bets")
    board = bets_scoreboard([dict(r) for r in chillsdb.responses_for_sender(user["id"])])
    yp, ap = board["you_points"], board["algo_points"]
    if yp > ap:
        score_line = f"You lead the algorithm {yp} to {ap}."
    elif ap > yp:
        score_line = f"The algorithm leads {ap} to {yp}."
    else:
        score_line = f"Level at {yp} to {ap}."
    responses = [dict(r) for r in chillsdb.responses_for_send(token)]
    for r in responses:
        r["mode"] = row["mode"]
        r["outcome"] = bet_outcome(r)
        r["rname"] = (r["respondent_name"] or "").strip() or "A friend"
        r["avatar_letter"] = avatar_of(r["respondent_name"] or "?")
    unrevealed = [r for r in responses if not r["revealed_at"]]
    revealed = [r for r in responses if r["revealed_at"]]
    gate_resp = unrevealed[0] if unrevealed else None
    got_n = sum(1 for r in revealed if r["experienced"])
    bet_you = sum(1 for r in revealed if r["outcome"] == "you")
    bet_algo = sum(1 for r in revealed if r["outcome"] == "algorithm")
    return t.TemplateResponse("reveal.html", {
        "request": req, "page": "reveal", **nav_context(req, user), "send": row,
        "gate_resp": gate_resp, "gate_remaining": len(unrevealed),
        "revealed": revealed, "got_n": got_n, "no_n": len(revealed) - got_n,
        "bet_you": bet_you, "bet_algo": bet_algo,
        "score_line": score_line, "base_url": req_base(req),
    })


@a.post("/reveal/{token}")
async def reveal_submit(req: Request, token: str):
    f = await req.form()
    closeness = int(f.get("closeness", 0) or 0)
    relationship = f.get("relationship", "")
    response_id = int(f.get("response_id", 0) or 0)
    row = chillsdb.get_send_by_token(token)
    user = chillsauth.get_current_user(req)
    resp = chillsdb.get_send_response(response_id) if response_id else None
    if (row and user and row["sender_user_id"] == user["id"]
            and resp and resp["send_token"] == token and not resp["revealed_at"]):
        chillsdb.reveal_send_response(response_id, closeness, relationship)
    if row and row["status"] == "watched":
        chillsdb.record_reveal_gate(token, closeness, relationship)
        log_ev("revealed", user=chillsauth.get_current_user(req), detail={
            "token": token, "closeness": closeness, "relationship": relationship,
            "experienced": bool(row["experienced"]), "outcome": bet_outcome(dict(row)) or "",
        })
    return RedirectResponse(f"/reveal/{token}", status_code=303)


# ═══════════════════════════════════════════════════
# V38 - CHILLS COMPATIBILITY (DUO)
# ═══════════════════════════════════════════════════

@a.get("/duo/new", response_class=HTMLResponse)
def duo_new(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    with chillsdb.get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM duo_pairs WHERE user_id=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
            (user["id"],),
        ).fetchone()
    row = existing or chillsdb.create_duo(user["id"])
    if not existing:
        log_ev("duo_created", user=user, detail={"token": row["token"]})

    # v49 duo intro: completed pairs under Results, pending under Waiting with
    # opened-the-link state. The active pending row is the persistent link.
    results = []
    waiting = []
    for d in chillsdb.duos_for_user(user["id"]):
        d = dict(d)
        d["avatar_letter"] = avatar_of(d["partner_name"] or "?")
        d["avatar_color"] = avatar_color(d["partner_name"] or str(d["id"]))
        d["date_str"] = datetime.utcfromtimestamp(d["completed_at"] or d["created_at"]).strftime("%b %d")
        d["opened"] = bool(d.get("opened_at"))
        if d["status"] == "completed":
            results.append(d)
        elif d["status"] == "pending":
            waiting.append(d)

    return t.TemplateResponse("duo_new.html", {
        "request": req, "page": "duo-intro", **nav_context(req, user), "duo": row,
        "duo_results": results, "duo_waiting": waiting,
    })


@a.get("/us/{token}", response_class=HTMLResponse)
def duo_recipient(req: Request, token: str):
    row = chillsdb.get_duo_by_token(token)
    if not row:
        return RedirectResponse("/")
    viewer = chillsauth.get_current_user(req)
    # opened-tracking for the duo intro waiting list. Only the partner's first
    # visit counts, not the initiator previewing their own link.
    if row["status"] == "pending" and not (viewer and viewer["id"] == row["user_id"]):
        chillsdb.mark_duo_opened(token)
        log_ev("duo_link_opened", user=viewer, detail={"token": token})
    initiator = chillsdb.get_user_by_id(row["user_id"])
    initiator_name = (initiator["pid"] if initiator and initiator["pid"] else "Someone") or "Someone"
    return t.TemplateResponse("duo_recipient.html", {
        "request": req, "page": "duo", **nav_context(req, viewer),
        "duo": row, "initiator_name": initiator_name,
        "initiator_letter": avatar_of(initiator_name), "guest": True,
    })


@a.get("/duo/{token}", response_class=HTMLResponse)
def duo_result(req: Request, token: str):
    row = chillsdb.get_duo_by_token(token)
    if not row or row["status"] != "completed":
        return RedirectResponse("/")
    initiator = chillsdb.get_user_by_id(row["user_id"])
    partner = chillsdb.get_user_by_id(row["partner_user_id"])
    a_name = (initiator["pid"] if initiator and initiator["pid"] else "You") or "You"
    b_name = (row["partner_name"] or "").strip() or "A friend"
    vid_sid = row["video_stimulus_id"] or ""
    viewer = chillsauth.get_current_user(req)
    me_side = None
    if viewer and viewer["id"] == row["user_id"]:
        me_side = "a"
    elif viewer and row["partner_user_id"] and viewer["id"] == row["partner_user_id"]:
        me_side = "b"
    jv = video_by_sid(vid_sid) or {}
    a_map = chillsdb.chills_by_video(row["user_id"])
    b_map = chillsdb.chills_by_video(row["partner_user_id"] or 0)
    try:
        va = json.loads(initiator["vector_json"] or "[]") if initiator else []
        vb = json.loads(partner["vector_json"] or "[]") if partner else []
    except Exception:
        va, vb = [], []
    res = chills_match(va, vb)
    if res is not None:
        pool = [stim_entry(j, 0.0) for j in res[2]]
    else:
        pool = []
        try:
            for e in (json.loads(initiator["top5_json"] or "[]") if initiator else [])[:5]:
                v0 = video_by_sid(e.get("stimulus_id", "")) or {}
                pool.append({"stimulus_id": e.get("stimulus_id", ""), "name": e.get("name", ""),
                             "url": e.get("url", "") or v0.get("url", ""), "dur": v0.get("dur", "")})
        except Exception:
            pool = []
    cv_list = []
    for e in pool[:5]:
        sid0 = e.get("stimulus_id", "")
        cv_list.append({**e, "a_chills": a_map.get(sid0), "b_chills": b_map.get(sid0),
                         "both_chills": bool(a_map.get(sid0)) and bool(b_map.get(sid0))})
    cv_match_count = sum(1 for e in cv_list if e["both_chills"])
    def _likely(u):
        return max(1, min(99, round(float((u["percentile"] if u else 0) or 0))))
    return t.TemplateResponse("duo_result.html", {
        "request": req, "page": "duo", **nav_context(req, viewer),
        "duo": row, "a_name": a_name,
        "a_pct": round(float(initiator["score"]) * 100) if initiator else 0,
        "a_percentile": initiator["percentile"] if initiator else 0,
        "b_name": b_name,
        "b_percentile": partner["percentile"] if partner else 0,
        "a_watched": chillsdb.has_watched(row["user_id"], vid_sid),
        "b_watched": chillsdb.has_watched(row["partner_user_id"] or 0, vid_sid),
        "joint_video": jv,
        "me_side": me_side,
        "letter_a": avatar_of(a_name), "letter_b": avatar_of(b_name),
        "a_likely_pct": _likely(initiator), "b_likely_pct": _likely(partner),
        "a_one_in": one_in_for(mean_p_of(initiator)) if initiator else 8,
        "b_one_in": one_in_for(mean_p_of(partner)) if partner else 8,
        "cv_list": cv_list, "cv_match_count": cv_match_count,
        "video_urls": [jv.get("url", "")], "base_url": req_base(req),
    })


# ═══════════════════════════════════════════════════
# V38 - CONTRIBUTE A VIDEO
# ═══════════════════════════════════════════════════

@a.get("/contribute", response_class=HTMLResponse)
def contribute_page(req: Request, sent: int = 0):
    user = chillsauth.get_current_user(req)
    return t.TemplateResponse("contribute.html", {
        "request": req, "page": "contribute", **nav_context(req, user), "sent": sent,
    })


@a.post("/contribute")
async def contribute_submit(req: Request):
    f = await req.form()
    url = (f.get("url") or "").strip()
    description = (f.get("description") or "").strip()
    user = chillsauth.get_current_user(req)
    if url or description:
        chillsdb.create_contribution(url, description, submitted_by=user["id"] if user else None)
        log_ev("contribution", user=user, detail={"url": url, "description": description})
        return RedirectResponse("/contribute?sent=1", status_code=303)
    return RedirectResponse("/contribute", status_code=303)


# ═══════════════════════════════════════════════════
# V38 - INFORMATIONAL PAGES
# ═══════════════════════════════════════════════════

@a.get("/about", response_class=HTMLResponse)
def about_page(req: Request):
    user = chillsauth.get_current_user(req)
    return t.TemplateResponse("about.html", {"request": req, "page": "about", **nav_context(req, user)})


@a.get("/method", response_class=HTMLResponse)
def method_page(req: Request):
    user = chillsauth.get_current_user(req)
    return t.TemplateResponse("method.html", {"request": req, "page": "method", **nav_context(req, user)})


@a.get("/terms", response_class=HTMLResponse)
def terms_page(req: Request):
    if not os.path.exists(os.path.join(b, "templates", "terms.html")):
        return HTMLResponse("Not here yet.", status_code=404)
    return t.TemplateResponse("terms.html", {"request": req, "page": "terms",
                                              **nav_context(req, chillsauth.get_current_user(req))})


@a.get("/privacy", response_class=HTMLResponse)
def privacy_page(req: Request):
    if not os.path.exists(os.path.join(b, "templates", "privacy.html")):
        return HTMLResponse("Not here yet.", status_code=404)
    return t.TemplateResponse("privacy.html", {"request": req, "page": "privacy",
                                                **nav_context(req, chillsauth.get_current_user(req))})


# ═══════════════════════════════════════════════════
# V38 - ADMIN CONSOLE
# ═══════════════════════════════════════════════════

@a.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(req: Request, error: int = 0):
    if chillsauth.is_admin(req):
        return RedirectResponse("/admin")
    return t.TemplateResponse("admin_login.html", {"request": req, "page": "admin-login", "error": error})


@a.post("/admin/login")
async def admin_login_submit(req: Request):
    f = await req.form()
    if not chillsauth.check_admin_login(f.get("email", ""), f.get("password", "")):
        return RedirectResponse("/admin/login?error=1", status_code=303)
    token = chillsdb.create_admin_session()
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(chillsauth.ADMIN_COOKIE, token, max_age=60*60*12, httponly=True, samesite="lax")
    return resp


@a.get("/admin/logout")
def admin_logout(req: Request):
    chillsdb.revoke_admin_session(req.cookies.get(chillsauth.ADMIN_COOKIE, ""))
    resp = RedirectResponse("/")
    resp.delete_cookie(chillsauth.ADMIN_COOKIE)
    return resp


@a.get("/admin/legacy", response_class=HTMLResponse)
def admin_console(req: Request, tab: str = "overview", sort: str = "created_at", page: int = 0):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")

    total_signups = chillsdb.count_users()
    total_paid = chillsdb.count_paid_users()
    scores = chillsdb.all_scores()
    with chillsdb.get_conn() as conn:
        sends_all = conn.execute("SELECT * FROM sends").fetchall()
    responded = [s for s in sends_all if s["status"] in ("watched", "revealed")]
    chills_rate = round(100 * sum(1 for s in responded if s["experienced"]) / len(responded)) if responded else 0
    shared_pct = round(100 * len(sends_all) / total_paid) if total_paid else 0
    reported_chills = chillsdb.count_sends_experienced()

    # today deltas for the tiles, midnight UTC
    now = datetime.utcnow()
    midnight = datetime(now.year, now.month, now.day).timestamp()
    signups_today = chillsdb.count_since("users", midnight)
    sends_today = chillsdb.count_since("sends", midnight)

    # cumulative growth series for the two charts, full history; the range
    # toggle (all time / last 30 days) slices client-side
    def cumulative_series(table):
        rows = chillsdb.daily_counts(table)
        days = []; totals = []; running = 0
        for r in rows:
            running += r["n"]
            days.append(r["day"]); totals.append(running)
        return {"days": days, "totals": totals}
    growth_signups = cumulative_series("users")
    growth_invited = cumulative_series("sends")

    # users table: load all, enrich, sort in python (needed for watched and
    # sent columns), then paginate
    per_page = 10
    all_rows = chillsdb.all_users(limit=100000)
    watched_map = chillsdb.watched_counts_map()
    sent_map = chillsdb.sent_counts_map()
    users_all = []
    for u in all_rows:
        d = dict(u)
        d["sent_n"] = sent_map.get(u["id"], 0)
        d["watched_n"] = watched_map.get(u["id"], 0)
        d["email"] = ""  # accounts land with the signup build; blank until then
        d["comments"] = [dict(c) for c in chillsdb.comments_by_user(u["id"])]
        d["created_str"] = datetime.utcfromtimestamp(u["created_at"]).strftime("%b %d")
        d["video_stimulus_name"] = u["stimulus_name"]
        # chills pill from their own reports: any yes -> yes, any report -> no, none -> dash
        if any(c.get("experienced") for c in d["comments"]):
            d["chills"] = 1
        elif d["comments"]:
            d["chills"] = 0
        else:
            d["chills"] = None
        users_all.append(d)
    keyers = {
        "created_at": lambda x: x["created_at"],
        "score": lambda x: x["score"] or 0,
        "watched": lambda x: x["watched_n"],
        "sent": lambda x: x["sent_n"],
    }
    users_all.sort(key=keyers.get(sort, keyers["created_at"]), reverse=True)
    total_pages = max(1, (len(users_all) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    users_view = users_all[page * per_page:(page + 1) * per_page]

    return t.TemplateResponse("admin.html", {
        "request": req, "page": "admin", "avatar_letter": "A", "notif_count": 0, "is_admin": True, "has_profile": False,
        "tab": tab, "sort": sort, "page_n": page, "total_pages": total_pages,
        "total_signups": total_signups, "total_paid": total_paid,
        "chills_rate": chills_rate, "shared_pct": shared_pct,
        "users": users_view, "scores_n": len(scores),
        "reported_chills": reported_chills, "shared_total": len(sends_all),
        "signups_today": signups_today, "sends_today": sends_today,
        "growth_signups": growth_signups, "growth_invited": growth_invited,
        "updated_str": datetime.utcnow().strftime("%b %d, %I:%M %p UTC"),
        "growth_json": json.dumps({"signups": growth_signups, "invited": growth_invited}),
    })


@a.get("/admin/export.csv")
def admin_export_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    rows = chillsdb.all_users(limit=100000)
    akeys = [x["k"] for x in qall()]
    out = [["signed_up","name","email","google","consented","beta_status","chills_score",
            "match_video","match_p","top5","paid","vector"] + akeys]
    for u in rows:
        try:
            aj = json.loads(u["answers_json"] or "{}")
        except Exception:
            aj = {}
        try:
            t5 = "; ".join(e.get("name","") for e in json.loads(u["top5_json"] or "[]"))
        except Exception:
            t5 = ""
        out.append([
            datetime.utcfromtimestamp(u["created_at"]).isoformat(),
            u["pid"], u["email"] or "", "yes" if (u["google_sub"] or "") else "",
            datetime.utcfromtimestamp(u["consented_at"]).isoformat() if u["consented_at"] else "",
            u["beta_status"] or "none",
            round(float(u["percentile"] or 0), 1),
            u["stimulus_name"], round(float(u["score"] or 0) * 100, 1), t5, bool(u["paid"]),
            u["vector_json"] or "",
        ] + [aj.get(k, "") for k in akeys])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_users.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/after.csv")
def admin_after_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT a.created_at, u.pid, u.email, a.stimulus_id, a.chills,
                      a.what_text, a.why_text
               FROM after_answers a LEFT JOIN users u ON u.id = a.user_id
               ORDER BY a.created_at DESC"""
        ).fetchall()
    out = [["time","name","email","stimulus_id","video","chills","what","why"]]
    for r in rows:
        cat = video_by_sid(r["stimulus_id"]) or {}
        out.append([
            datetime.utcfromtimestamp(r["created_at"]).isoformat(),
            r["pid"] or "", r["email"] or "", r["stimulus_id"], cat.get("name",""),
            "yes" if r["chills"] else "no", r["what_text"] or "", r["why_text"] or "",
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_after_answers.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/events.csv")
def admin_events_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    rows = chillsdb.all_events()
    out = [["time", "user_id", "pid", "event", "detail"]]
    for e in rows:
        out.append([
            datetime.utcfromtimestamp(e["created_at"]).isoformat(),
            e["user_id"] if e["user_id"] is not None else "",
            e["pid"], e["event"], e["detail"],
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_events.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/duo.csv")
def admin_duo_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT d.created_at, d.completed_at, u.pid, u.email, d.partner_name,
                      d.status, d.match_pct, d.video_stimulus_name, d.token
               FROM duo_pairs d LEFT JOIN users u ON u.id = d.user_id
               ORDER BY d.created_at DESC"""
        ).fetchall()
    out = [["created","completed","initiator","initiator_email","partner","status","match_pct","joint_video","token"]]
    for r in rows:
        out.append([
            datetime.utcfromtimestamp(r["created_at"]).isoformat(),
            datetime.utcfromtimestamp(r["completed_at"]).isoformat() if r["completed_at"] else "",
            r["pid"] or "", r["email"] or "", r["partner_name"] or "",
            r["status"] or "", round(float(r["match_pct"] or 0), 1),
            r["video_stimulus_name"] or "", r["token"],
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_duo.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/sends.csv")
def admin_sends_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT s.created_at, s.watched_at, s.revealed_at, u.pid, u.email,
                      s.recipient_name, s.stimulus_name, s.mode, s.status,
                      s.experienced, s.closeness, s.relationship, s.description, s.token
               FROM sends s LEFT JOIN users u ON u.id = s.sender_user_id
               ORDER BY s.created_at DESC"""
        ).fetchall()
    out = [["created","sender","sender_email","recipient","video","mode","status",
            "experienced","closeness","relationship","description","watched","revealed","token"]]
    for r in rows:
        exp = "" if r["experienced"] is None else ("yes" if r["experienced"] else "no")
        out.append([
            datetime.utcfromtimestamp(r["created_at"]).isoformat(),
            r["pid"] or "", r["email"] or "", r["recipient_name"] or "",
            r["stimulus_name"] or "", r["mode"] or "", r["status"] or "",
            exp, r["closeness"] if r["closeness"] is not None else "",
            r["relationship"] or "", r["description"] or "",
            datetime.utcfromtimestamp(r["watched_at"]).isoformat() if r["watched_at"] else "",
            datetime.utcfromtimestamp(r["revealed_at"]).isoformat() if r["revealed_at"] else "",
            r["token"],
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_sends.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/watches.csv")
def admin_watches_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT w.watched_at, u.pid, u.email, w.stimulus_id
               FROM video_watches w LEFT JOIN users u ON u.id = w.user_id
               ORDER BY w.watched_at DESC"""
        ).fetchall()
    out = [["time","name","email","stimulus_id","video"]]
    for r in rows:
        cat = video_by_sid(r["stimulus_id"]) or {}
        out.append([
            datetime.utcfromtimestamp(r["watched_at"]).isoformat(),
            r["pid"] or "", r["email"] or "", r["stimulus_id"], cat.get("name",""),
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_watches.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/comments.csv")
def admin_comments_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT c.created_at, c.author, c.stimulus_id, c.experienced, c.text, u.email
               FROM video_comments c LEFT JOIN users u ON u.id = c.user_id
               ORDER BY c.created_at DESC"""
        ).fetchall()
    out = [["time","author","email","stimulus_id","video","chills","text"]]
    for r in rows:
        cat = video_by_sid(r["stimulus_id"]) or {}
        out.append([
            datetime.utcfromtimestamp(r["created_at"]).isoformat(),
            r["author"] or "", r["email"] or "", r["stimulus_id"], cat.get("name",""),
            "yes" if r["experienced"] else "no", r["text"] or "",
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_comments.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


@a.get("/admin/contributions.csv")
def admin_contributions_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    with chillsdb.get_conn() as conn:
        rows = conn.execute(
            """SELECT c.created_at, c.url, c.description, c.status, u.pid, u.email
               FROM contributions c LEFT JOIN users u ON u.id = c.submitted_by
               ORDER BY c.created_at DESC"""
        ).fetchall()
    out = [["time","name","email","url","status","description"]]
    for r in rows:
        out.append([
            datetime.utcfromtimestamp(r["created_at"]).isoformat(),
            r["pid"] or "", r["email"] or "", r["url"] or "", r["status"] or "", r["description"] or "",
        ])
    csv_text = "\n".join(",".join('"' + str(c).replace('"', '""') + '"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_contributions.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


# ═══════════════════════════════════════════════════
# ADMIN CONSOLE v2 (Felix's rewire admin, pixel to pixel)
# ═══════════════════════════════════════════════════

@a.get("/admin", response_class=HTMLResponse)
def admin_console_v2(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    # served raw, not through Jinja, so Felix's markup and JS stay untouched
    with open(os.path.join(b, "templates", "admin_console.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


def _admin_json_guard(req: Request):
    if not chillsauth.is_admin(req):
        return JSONResponse({"ok": False, "error": "auth"}, status_code=401)
    return None


ADMIN_SETTING_KEYS = ("beta_weekly", "link_expiry", "grant_subject", "grant_message")
ADMIN_SETTING_DEFAULTS = {
    "beta_weekly": "0",
    "link_expiry": "They never expire",
    "grant_subject": "Your access to Edge is ready",
    "grant_message": "Hi {name},\n\nYour access to Edge is ready. Open this link on your phone and it signs you in:\n{link}\n\nF\u00e9lix",
}


def _edge_link(code: str) -> str:
    return "app.rewire.bio/go/" + (code or "")


@a.get("/admin/data")
def admin_data(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    users = chillsdb.users_all()
    tags = chillsdb.tags_map()
    notes = chillsdb.notes_map()
    sent_map = chillsdb.sent_counts_map()
    duo_map = chillsdb.duo_counts_map()
    last_map = chillsdb.last_active_map()
    afters_by_user = {}
    for r in chillsdb.after_answers_all():
        afters_by_user.setdefault(r["user_id"], []).append(r)
    names_by_id = {}
    people = []
    for u in users:
        rep_rows = afters_by_user.get(u["id"], [])
        reports = []
        for r in rep_rows:
            cat = video_by_sid(r["stimulus_id"]) or {}
            reports.append({
                "video": cat.get("name", "") or r["stimulus_id"],
                "chills": bool(r["chills"]),
                "what": r["what_text"] or "",
                "why": r["why_text"] or "",
                "date": r["created_at"],
            })
        matched_sid = u["stimulus_id"] or ""
        chills = any(r["chills"] for r in rep_rows if r["stimulus_id"] == matched_sid)
        status = {"none": "none", "listed": "requested", "granted": "granted"}.get(u["beta_status"] or "none", "none")
        edge = {"status": status, "opened": False, "signedIn": None, "sessions": [], "protocols": []}
        if status == "requested":
            edge["requested"] = u["beta_requested_at"] or u["created_at"]
        if status == "granted":
            edge["granted"] = u["edge_granted_at"] or u["created_at"]
            edge["link"] = _edge_link(u["edge_code"])
        name = (u["pid"] or "").strip() or ((u["email"] or "").split("@")[0]) or "Anonymous"
        names_by_id[u["id"]] = name
        people.append({
            "id": u["id"],
            "name": name,
            "email": u["email"] or "",
            "created": u["created_at"],
            "test": bool(u["paid"]) and bool(matched_sid),
            "chills": bool(chills),
            "score": round(float(u["percentile"] or 0)),
            "video": u["stimulus_name"] or "",
            "reports": reports,
            "invites": sent_map.get(u["id"], 0),
            "compat": duo_map.get(u["id"], 0),
            "tags": tags.get(u["id"], []),
            "notes": notes.get(u["id"], []),
            "edge": edge,
            "lastActive": last_map.get(u["id"]) or u["created_at"],
        })
    sends = chillsdb.sends_all()
    mode_by_token = {x["token"]: x["mode"] for x in sends}
    sb = bets_scoreboard([{"experienced": r["experienced"], "mode": mode_by_token.get(r["send_token"], "picked")}
                          for r in chillsdb.responses_all()])
    duos = chillsdb.duos_all()
    subs = []
    for c in chillsdb.all_contributions(limit=1000):
        if (c["status"] or "pending") != "pending":
            continue
        subs.append({
            "id": c["id"],
            "who": names_by_id.get(c["submitted_by"], "Anonymous"),
            "what": " ".join(x for x in [(c["description"] or "").strip(), (c["url"] or "").strip()] if x),
            "date": c["created_at"],
        })
    settings = dict(ADMIN_SETTING_DEFAULTS)
    settings.update({k: v for k, v in chillsdb.all_settings().items() if k in ADMIN_SETTING_KEYS})
    return JSONResponse({
        "now": time.time(),
        "people": people,
        "flags": [],
        "screened": 0,
        "subs": subs,
        "games": {
            "compat_sent": len(duos),
            "compat_completed": sum(1 for d in duos if d["status"] == "completed"),
            "rounds": sb["rounds_played"],
            "rounds_won": sb["you_points"],
        },
        "settings": settings,
        "admin_email": chillsauth.ADMIN_EMAIL or "admin",
    })


@a.post("/admin/grant")
async def admin_grant(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    user = chillsdb.get_user_by_id(int(d.get("user_id") or 0))
    if not user:
        return JSONResponse({"ok": False, "error": "No such user."})
    code = chillsdb.grant_edge(user["id"])
    log_ev("admin_edge_granted", user=user)
    return JSONResponse({"ok": True, "link": _edge_link(code)})


@a.post("/admin/revoke")
async def admin_revoke(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    user = chillsdb.get_user_by_id(int(d.get("user_id") or 0))
    if not user:
        return JSONResponse({"ok": False, "error": "No such user."})
    chillsdb.revoke_edge(user["id"])
    log_ev("admin_edge_revoked", user=user)
    return JSONResponse({"ok": True})


@a.post("/admin/grant-all")
async def admin_grant_all(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    n = chillsdb.grant_all_edge()
    log_ev("admin_edge_granted_all", detail={"n": n})
    return JSONResponse({"ok": True, "n": n})


@a.post("/admin/tag")
async def admin_tag(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    chillsdb.add_tag(int(d.get("user_id") or 0), d.get("tag") or "")
    return JSONResponse({"ok": True})


@a.post("/admin/note")
async def admin_note(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    chillsdb.add_admin_note(int(d.get("user_id") or 0), d.get("text") or "")
    return JSONResponse({"ok": True})


@a.post("/admin/invite")
async def admin_invite(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    name = (d.get("name") or "").strip()
    email = (d.get("email") or "").strip().lower()
    if not name or not email:
        return JSONResponse({"ok": False, "error": "Name and email, please"})
    user = chillsdb.get_user_by_email(email)
    if not user:
        user = chillsdb.create_account_user(email)
    if name and (user["pid"] or "") != name:
        with chillsdb.get_conn() as conn:
            conn.execute("UPDATE users SET pid=? WHERE id=?", (name, user["id"]))
    tag = (d.get("tag") or "").strip()
    if tag:
        chillsdb.add_tag(user["id"], tag)
    note = (d.get("note") or "").strip()
    if note:
        chillsdb.add_admin_note(user["id"], note)
    code = chillsdb.grant_edge(user["id"])
    log_ev("admin_edge_invited", detail={"user_id": user["id"]})
    return JSONResponse({"ok": True, "user_id": user["id"], "link": _edge_link(code)})


@a.post("/admin/sub/{cid}/approve")
async def admin_sub_approve(req: Request, cid: int):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    chillsdb.set_contribution_status(cid, "approved")
    log_ev("admin_sub_approved", detail={"cid": cid})
    return JSONResponse({"ok": True})


@a.post("/admin/sub/{cid}/decline")
async def admin_sub_decline(req: Request, cid: int):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    chillsdb.set_contribution_status(cid, "declined")
    log_ev("admin_sub_declined", detail={"cid": cid})
    return JSONResponse({"ok": True})


@a.post("/admin/settings")
async def admin_settings_save(req: Request):
    guard = _admin_json_guard(req)
    if guard:
        return guard
    d = await req.json()
    for k in ADMIN_SETTING_KEYS:
        if k in d:
            chillsdb.set_setting(k, str(d[k]))
    return JSONResponse({"ok": True})


# ═══════════════════════════════════════════════════
# FEEDBACK ROUTES (original detailed questionnaire)
# ═══════════════════════════════════════════════════

@a.get("/feedback", response_class=HTMLResponse)
def feedback(req: Request, id: str = "", stimulus_id: str = "", url: str = "", score: float = 0.0, stimulus_name: str = "", session_id: str = "", send_token: str = ""):
    return t.TemplateResponse("feedback.html", {
        "request": req, "id": id, "stimulus_id": stimulus_id, "url": url,
        "score": score, "stimulus_name": stimulus_name, "class_idx": -1,
        "session_id": session_id, "send_token": send_token,
    })

@a.post("/submit", response_class=HTMLResponse)
async def submit(req: Request,
    id: str = Form(...),
    stimulus_id: str = Form(...),
    url: str = Form(""),
    experienced: str = Form(...),
    chills_amount: int = Form(0),
    chills_length: int = Form(0),
    chills_waves: int = Form(0),
    description: str = Form(""),
    email: str = Form(""),
    prolific_id: str = Form(""),
    send_token: str = Form(""),
):
    p = "/data/logs.csv"
    Hh = [
        "ts","participant_id","email","prolific_id","stimulus_id","url",
        "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10",
        "description"
    ]
    is_new = not os.path.exists(p)
    clean_description = description.replace("\r\n", "\n").strip()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if is_new: w.writerow(Hh)
        w.writerow([
            datetime.utcnow().isoformat(), id, email, prolific_id, stimulus_id, url,
            experienced, chills_amount, chills_length, chills_waves,
            clean_description
        ])

    log_ev("chills_feedback", user=chillsauth.get_current_user(req), pid=id, detail={
        "stimulus_id": stimulus_id, "experienced": experienced, "intensity": chills_amount,
        "length": chills_length, "waves": chills_waves, "description": clean_description,
        "send_token": send_token,
    })

    # picked-mode Send Chills threads its token explicitly through the form;
    # algo-mode has no send_token here, it closes via the recipient's own
    # pending_send_token instead (set when they finished the questionnaire).
    # every report lands in send_responses; the legacy send row only takes
    # the first answer, so a second friend never overwrites the first.
    reporter = chillsauth.get_current_user(req)
    if send_token and chillsdb.get_send_by_token(send_token):
        chillsdb.create_send_response(send_token, experienced == "yes", intensity=chills_amount,
                                       chills_length=chills_length, chills_waves=chills_waves,
                                       description=clean_description, respondent_name=id,
                                       respondent_user_id=reporter["id"] if reporter else None)
        if chillsdb.get_send_by_token(send_token)["status"] == "sent":
            chillsdb.record_send_response(send_token, experienced == "yes",
                                           intensity=chills_amount, recipient_name=id,
                                           chills_length=chills_length, chills_waves=chills_waves,
                                           description=clean_description)
    else:
        user = chillsauth.get_current_user(req)
        if user and user["pending_send_token"] and user["stimulus_id"] == stimulus_id:
            chillsdb.record_send_response(user["pending_send_token"], experienced == "yes",
                                           intensity=chills_amount, recipient_name=id,
                                           chills_length=chills_length, chills_waves=chills_waves,
                                           description=clean_description)
            chillsdb.set_pending_send_token(user["token"], "")

    done_user = chillsauth.get_current_user(req)
    return t.TemplateResponse("done.html", {"request": req, "id": id, "email": email,
                                             **nav_context(req, done_user),
                                             "has_profile": _user_has_profile(done_user)})


# ═══════════════════════════════════════════════════
# EXISTING UTILITY ROUTES
# ═══════════════════════════════════════════════════

@a.get("/download-logs")
def download_logs():
    p = "/data/logs.csv"
    if not os.path.exists(p):
        return HTMLResponse("No logs yet \u2014 /data/logs.csv not found", status_code=404)
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    return FileResponse(p, media_type="text/csv", filename="logs.csv", headers=headers)


# ═══════════════════════════════════════════════════
# DEBUG ROUTES (all preserved)
# ═══════════════════════════════════════════════════

@a.get("/_debug/logs_head")
def logs_head(n: int = 5):
    p = "/data/logs.csv"
    if not os.path.exists(p):
        return {"exists": False, "size": 0, "lines": []}
    with open(p, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return {"exists": True, "size": os.path.getsize(p), "rows": len(lines)-1, "tail": lines[-min(n, len(lines)):]}

@a.get("/_debug/disk")
def disk():
    p = "/data"
    exists = os.path.isdir(p)
    listing = []
    if exists:
        try:
            listing = sorted(os.listdir(p))
        except Exception as e:
            listing = [f"error: {e}"]
    return {"mounted": exists, "path": p, "ls": listing}

@a.get("/_debug/onnx_status")
def onnx_status():
    cols = list(G.columns) if len(G) > 0 else []
    outs = [o.name for o in sess.get_outputs()]
    oshapes = []
    try:
        oshapes = [getattr(y, "shape", None) for y in sess.run(outs, {inn: to40X([0]*len(FEATURES))})]
    except Exception:
        pass

    unmatched_names = [STIM[i] for i in range(len(STIM)) if i not in IDX]

    return {
        "onnx_called": P["onnx_called"],
        "in_shape": P.get("in_shape"),
        "out_shape": P.get("out_shape"),
        "out_names": P.get("out_names", outs),
        "chosen_head_idx": P.get("chosen_head_idx"),
        "chosen_head_name": P.get("chosen_head_name"),
        "argmax": P["argmax"],
        "probs_len": len(P["probs"]) if P["probs"] else 0,
        "stimuli_cols": cols[:12],
        "rows_in_stimuli": len(G),
        "matched": len(IDX),
        "unmatched": unmatched_names,
        "live_out_shapes": oshapes,
    }

@a.get("/_debug/last_error")
def last_error():
    return E

@a.get("/_debug/last_probs")
def last_probs():
    return {"argmax": P.get("argmax"), "probs": P.get("probs"), "in_shape": P.get("in_shape"), "out_shape": P.get("out_shape")}

@a.get("/_debug/last_top5")
def last_top5():
    return {"top5": P.get("last_top5", [])}

@a.get("/_debug/stim_match")
def stim_match():
    out = []
    for i, sname in enumerate(STIM):
        m2 = IDX.get(i, None)
        if m2 is None or m2 >= len(G):
            out.append({"stim_idx": i, "stim_name": sname, "matched": False})
        else:
            r = G.iloc[m2]
            out.append({
                "stim_idx": i, "stim_name": sname, "matched": True, "csv_row": int(m2),
                "csv_name": str(r.get(CSV_NAME,"")), "csv_url": str(r.get(CSV_URL,""))
            })
    return {"total": len(STIM), "matched": sum(1 for z in out if z['matched']), "items": out,
            "unmatched": [{"stim": u["stim"], "canon": u["canon"]} for u in UNM]}

@a.get("/_debug/feature_wire")
def feature_wire():
    return FW

@a.get("/_debug/avoid")
def avoid_debug(threshold: float = 0.5):
    probs = P.get("probs") or []
    avoid = []
    if probs and len(probs) == len(STIM):
        for i, pr in enumerate(probs):
            try:
                if float(pr) < float(threshold):
                    avoid.append(STIM[i])
            except Exception:
                continue
    return {"threshold": float(threshold), "count": len(avoid), "avoid": avoid}


# ═══════════════════════════════════════════════════
# 404
# ═══════════════════════════════════════════════════
@a.exception_handler(404)
async def not_found(req: Request, exc):
    if req.url.path.startswith("/_debug") or req.url.path.startswith("/api"):
        return JSONResponse({"error": "not found"}, status_code=404)
    user = chillsauth.get_current_user(req)
    return t.TemplateResponse("404.html", {
        "request": req, "page": "404", **nav_context(req, user),
    }, status_code=404)

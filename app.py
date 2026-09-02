import sitecustomize
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import csv, hashlib, json, os, re, secrets, time, unicodedata
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
b = os.path.dirname(__file__)
t = Jinja2Templates(directory=os.path.join(b, "templates"))

ff = os.path.join(b, "minimal_features.json")
pf = os.path.join(b, "preprocessor_minimal.joblib")
mf = os.path.join(b, "final_global_mlp.onnx")
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

def qall():
    u = []; u.extend(Q); u.extend(qdemo()); return u


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

def ag(x):
    x = str(x or "").strip()
    try: return float(x)
    except:
        y = re.sub(r"\s+","",x)
        m = re.match(r"^(\d+)[\-\u2013](\d+)$", y)
        if m: return (float(m.group(1)) + float(m.group(2))) / 2.0
        m = re.match(r"^(\d+)\+$", y)
        if m: return float(m.group(1)) + 5.0
        d = {"18-24":21,"25-34":30,"35-44":40,"45-54":50,"55-64":60,"65+":70}
        return float(d.get(x,0))

def build_answer_maps(H):
    A = {}
    for k,v in H.items(): A[nkey(k)] = v
    return A

def map_answers_to_features(H):
    HN = build_answer_maps(H)
    m = []; pairs = []; miss = []
    for fk in FEATURES:
        if fk == "Age":
            v = ag(H.get("Age",""))
            m.append(v); pairs.append([fk, v, "Age"])
        elif fk.lower() == "stimulus":
            m.append(0.0); pairs.append([fk, 0.0, "Stimulus"])
        else:
            vv = None
            if fk in H: vv = H.get(fk, None)
            if vv is None: vv = HN.get(nkey(fk), None)
            if vv is None and "_" in fk: vv = HN.get(nkey(fk.replace("_"," ")), None)
            if vv is None and "-" in fk: vv = HN.get(nkey(fk.replace("-"," ")), None)
            if vv is None:
                miss.append(fk); m.append(0.0); pairs.append([fk, 0.0, "default0"])
            else:
                try: v = float(vv)
                except: v = ag(vv)
                m.append(v); pairs.append([fk, v, "mapped"])
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


CHILLS_HEAD_ENV = os.getenv("REWIRE_CHILLS_HEAD_INDEX")  
CHILLS_NAME_HINTS = ("chills", "chills_bin", "prob_chills", "head0")

def _choose_chills_head_index(outs: List[str]) -> int:
    if CHILLS_HEAD_ENV is not None:
        try:
            i = int(CHILLS_HEAD_ENV)
            if 0 <= i < len(outs):
                return i
        except Exception:
            pass
    ln = [str(x).lower() for x in outs]
    for hint in CHILLS_NAME_HINTS:
        for i, n in enumerate(ln):
            if hint in n:
                return i
    return 0

def _extract_from_probabilities_struct(prob_output: Any) -> Optional[np.ndarray]:
    try:
        if not hasattr(prob_output, "__len__") or len(prob_output) != len(STIM):
            return None
        if isinstance(prob_output[0], dict):
            chills = []
            for row in prob_output:
                found = None
                for key in ("Chills_bin","chills_bin","chills","0","head0","H0"):
                    if key in row:
                        val = row[key]
                        if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                            found = float(val[1])
                        elif isinstance(val, (int, float)):
                            found = float(val)
                        break
                if found is None:
                    if "0" in row and isinstance(row["0"], (list, tuple, np.ndarray)) and len(row["0"]) >= 2:
                        found = float(row["0"][1])
                    elif "0" in row and isinstance(row["0"], (int, float)):
                        found = float(row["0"])
                    else:
                        for v in row.values():
                            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                                found = float(v[1]); break
                            if isinstance(v, (int, float)):
                                found = float(v); break
                chills.append(found if found is not None else 0.0)
            return np.asarray(chills, dtype=np.float32)
        if isinstance(prob_output[0], (list, tuple, np.ndarray)):
            arr = np.asarray(prob_output)
            if arr.ndim == 2 and arr.shape[0] == len(STIM):
                return arr[:, 1].astype(np.float32)
            if arr.ndim == 3 and arr.shape[:2] == (len(STIM), 2):
                return arr[:, 1, 0].astype(np.float32)
        return None
    except Exception:
        return None

def topk(v, k=1, pid=""):
    X = to40X(v)
    out_defs = sess.get_outputs()
    outs = [o.name for o in out_defs]
    yl = sess.run(outs, {inn: X})

    p = None
    used_idx = None
    used_name = None

    prob_idx = None
    for i, n in enumerate(outs):
        if "probabilities" in str(n).lower():
            prob_idx = i
            break

    if prob_idx is not None:
        y = yl[prob_idx]
        if isinstance(y, (list, tuple)):
            try:
                head0 = y[0] 
            except Exception as ex:
                raise RuntimeError(f"'probabilities' is a sequence but empty/invalid: type={type(y)}") from ex
            arr = np.asarray(head0)
            if arr.ndim == 2 and arr.shape[0] == len(STIM) and arr.shape[1] >= 2:
                p = arr[:, 1].astype(np.float32)  
            elif arr.ndim == 1 and arr.shape[0] == len(STIM):
                p = arr.astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected shape for CHILLS head0: {arr.shape}; expected (40,2) or (40,).")
        else:
            p = _extract_from_probabilities_struct(y)
            if p is None:
                arr = np.asarray(y)
                if arr.ndim == 2 and arr.shape[0] == len(STIM) and arr.shape[1] >= 2:
                    p = arr[:, 1].astype(np.float32)
                elif arr.ndim == 1 and arr.shape[0] == len(STIM):
                    p = arr.astype(np.float32)
                else:
                    raise RuntimeError(f"Could not parse 'probabilities' output. shape={getattr(y,'shape',None)}")
        used_idx = prob_idx
        used_name = outs[prob_idx]

    if p is None:
        hi = _choose_chills_head_index(outs)
        y = yl[hi]
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[0] == len(STIM) and arr.shape[1] >= 2:
            p = arr[:, 1].astype(np.float32)
        elif arr.ndim == 1 and arr.shape[0] == len(STIM):
            p = arr.astype(np.float32)
        else:
            p = _extract_from_probabilities_struct(y)
            if p is None:
                raise RuntimeError(
                    f"Could not resolve CHILLS probabilities. chosen_head_idx={hi}, head_shape={getattr(y,'shape',None)}, outs={outs}"
                )
        used_idx = hi
        used_name = outs[hi]

    eps = (np.arange(len(STIM)) * 1e-9).astype(np.float32)
    p = p + eps
    if np.max(p) - np.min(p) < 1e-6:
        z = (p - np.mean(p)) / (np.std(p) + 1e-9)
        idx = np.argsort(-z)[:k]
    else:
        idx = np.argsort(-p)[:k]

    try:
        P["onnx_called"] = True
        P["in_shape"] = tuple(X.shape)
        P["out_shape"] = [getattr(z, "shape", None) if hasattr(z, "shape") else None for z in yl]
        P["out_names"] = outs
        P["probs"] = [float(z) for z in p.tolist()]
        P["argmax"] = int(idx[0]) if len(idx) else None
        P["chosen_head_idx"] = used_idx
        P["chosen_head_name"] = used_name
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
# ROUTES
# ═══════════════════════════════════════════════════

@a.get("/", response_class=HTMLResponse)
def index(req: Request, send: str = "", us: str = ""):
    if chillsauth.is_admin(req) and not send and not us:
        return RedirectResponse("/admin")
    q2 = [x for x in Q if x["k"] != "Age"]
    resp = t.TemplateResponse("index.html", {"request": req, "DEMO": qdemo(), "QS": q2})
    if send:
        resp.set_cookie("pending_link", json.dumps({"type": "send", "token": send}),
                         max_age=3600, httponly=True, samesite="lax")
    elif us:
        resp.set_cookie("pending_link", json.dumps({"type": "duo", "token": us}),
                         max_age=3600, httponly=True, samesite="lax")
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
        for x in qall(): H[x["k"]] = f.get(x["n"], "")
        FW["raw_answers"] = H
        m = map_answers_to_features(H)

        best5 = topk(m, 5, pid=pid)
        P["last_top5"] = best5[:]

        best = _choose_from_ties(best5, pid, built_vec=FW.get("built_vector", [])) if best5 else {
            "score": 0.0, "stimulus_id": "", "url": "", "name": "", "desc": "", "dur": "", "cap": ""
        }

        # store in session
        sid = _new_session()
        SESSIONS[sid]["pid"] = pid
        SESSIONS[sid]["stimulus"] = best
        SESSIONS[sid]["top5"] = best5
        SESSIONS[sid]["vector"] = FW.get("built_vector", [])
        SESSIONS[sid]["paid"] = False

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

    share_url = f"https://chillstv.com/p/{sid[:8]}"

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

    # persist a hub profile for this visitor (cookie-based, no signup required)
    visitor_token = req.cookies.get(chillsauth.VISITOR_COOKIE, "")
    user = chillsdb.get_user_by_token(visitor_token)
    if not user:
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
    vector_json = json.dumps(sess_data.get("vector", []))
    score01 = float(stim.get("score", 0.0))
    chillsdb.update_user_match(
        visitor_token, stim.get("stimulus_id",""), stim.get("stim_name", stim.get("name","")),
        stim.get("url",""), score01, 0.0, paid=True, top5_json=top5_json, vector_json=vector_json,
    )
    chillsdb.set_percentile(visitor_token, chillsdb.percentile_for_score(score01))

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
            if duo_row and duo_row["status"] == "pending":
                initiator = chillsdb.get_user_by_id(duo_row["user_id"])
                init_vec = json.loads(initiator["vector_json"] or "[]") if initiator else []
                match_pct = cosine_match_pct(init_vec, sess_data.get("vector", []))
                chillsdb.complete_duo(
                    pending["token"], user["id"] if user else 0, pid or "A friend", match_pct,
                    stim.get("stimulus_id",""), stim.get("stim_name", stim.get("name","")),
                )
                dest = f"/duo/{pending['token']}"

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

@a.get("/hub", response_class=HTMLResponse)
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
    return t.TemplateResponse("hub.html", {
        "request": req, "page": "hub", **nav_context(req, user), "videos": videos,
        "recent_sends": recent, "sent_count": sent_count, "hit_count": hit_count,
        "hit_rate": hit_rate,
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
    return t.TemplateResponse("video.html", {
        "request": req, "page": "video", **nav_context(req, user),
        "video": v, "embed_url": _to_embed_url(v["url"]), "comments": comments,
        "pid": (user["pid"] if user else "") or "Anonymous",
    })


@a.post("/video/{sid}/report")
async def video_report(req: Request, sid: str):
    f = await req.form()
    experienced = (f.get("experienced") or "") == "yes"
    text = (f.get("text") or "").strip()
    user = chillsauth.get_current_user(req)
    author = (user["pid"] if user and user["pid"] else "Anonymous") or "Anonymous"
    if text:
        chillsdb.add_video_comment(sid, text, experienced, author=author)

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
    for s in sends:
        s["avatar_letter"] = avatar_of(s["recipient_name"] or "?")
        s["avatar_color"] = avatar_color(s["recipient_name"] or str(s["id"]))
    ready = [s for s in sends if s["status"] == "watched"]
    unopened = [s for s in sends if s["status"] == "sent"]
    done = [s for s in sends if s["status"] == "revealed"]
    hit_n = sum(1 for s in done if s["experienced"])
    hit_rate = round(100 * hit_n / len(done)) if done else 0
    return t.TemplateResponse("bets.html", {
        "request": req, "page": "bets", **nav_context(req, user),
        "ready": ready, "unopened": unopened, "done": done, "hit_rate": hit_rate,
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
    })


@a.post("/send/create")
async def send_create(req: Request):
    user = chillsauth.get_current_user(req)
    if not user:
        return RedirectResponse("/")
    f = await req.form()
    mode = f.get("mode", "picked")
    if mode == "algo":
        row = chillsdb.create_send(user["id"], "", "", "", mode="algo")
    else:
        sid = f.get("stimulus_id", "")
        v = video_by_sid(sid) or {}
        row = chillsdb.create_send(user["id"], sid, v.get("name", ""), v.get("url", ""), mode="picked")
    return RedirectResponse(f"/send/{row['token']}", status_code=303)


@a.get("/send/{token}", response_class=HTMLResponse)
def send_show(req: Request, token: str):
    user = chillsauth.get_current_user(req)
    row = chillsdb.get_send_by_token(token)
    if not user or not row or row["sender_user_id"] != user["id"]:
        return RedirectResponse("/bets")
    return t.TemplateResponse("send_link.html", {
        "request": req, "page": "bet-intro", **nav_context(req, user), "send": row,
    })


@a.get("/b/{token}", response_class=HTMLResponse)
def bet_view(req: Request, token: str):
    row = chillsdb.get_send_by_token(token)
    if not row:
        return RedirectResponse("/")
    sender = chillsdb.get_user_by_id(row["sender_user_id"])
    sender_name = (sender["pid"] if sender and sender["pid"] else "Someone") or "Someone"
    recipient = chillsauth.get_current_user(req)
    return t.TemplateResponse("bet_view.html", {
        "request": req, "page": "bet-view", **nav_context(req, recipient),
        "send": row, "sender_name": sender_name,
        "recipient_pid": (recipient["pid"] if recipient else "") or "Anonymous",
    })


@a.post("/b/{token}/respond")
async def bet_respond(req: Request, token: str):
    f = await req.form()
    experienced = (f.get("experienced") or "") == "yes"
    row = chillsdb.get_send_by_token(token)
    if row and row["status"] == "sent":
        chillsdb.record_send_response(token, experienced)
    return RedirectResponse(f"/b/{token}", status_code=303)


@a.get("/reveal/{token}", response_class=HTMLResponse)
def reveal_gate(req: Request, token: str):
    user = chillsauth.get_current_user(req)
    row = chillsdb.get_send_by_token(token)
    if not user or not row or row["sender_user_id"] != user["id"]:
        return RedirectResponse("/bets")
    return t.TemplateResponse("reveal.html", {
        "request": req, "page": "reveal", **nav_context(req, user), "send": row,
    })


@a.post("/reveal/{token}")
async def reveal_submit(req: Request, token: str):
    f = await req.form()
    closeness = int(f.get("closeness", 0) or 0)
    relationship = f.get("relationship", "")
    row = chillsdb.get_send_by_token(token)
    if row and row["status"] == "watched":
        chillsdb.record_reveal_gate(token, closeness, relationship)
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
    return t.TemplateResponse("duo_new.html", {
        "request": req, "page": "duo-intro", **nav_context(req, user), "duo": row,
    })


@a.get("/us/{token}", response_class=HTMLResponse)
def duo_recipient(req: Request, token: str):
    row = chillsdb.get_duo_by_token(token)
    if not row:
        return RedirectResponse("/")
    initiator = chillsdb.get_user_by_id(row["user_id"])
    initiator_name = (initiator["pid"] if initiator and initiator["pid"] else "Someone") or "Someone"
    return t.TemplateResponse("duo_recipient.html", {
        "request": req, "page": "duo", **nav_context(req, chillsauth.get_current_user(req)),
        "duo": row, "initiator_name": initiator_name,
    })


@a.get("/duo/{token}", response_class=HTMLResponse)
def duo_result(req: Request, token: str):
    row = chillsdb.get_duo_by_token(token)
    if not row or row["status"] != "completed":
        return RedirectResponse("/")
    initiator = chillsdb.get_user_by_id(row["user_id"])
    a_name = (initiator["pid"] if initiator and initiator["pid"] else "You") or "You"
    return t.TemplateResponse("duo_result.html", {
        "request": req, "page": "duo", **nav_context(req, chillsauth.get_current_user(req)),
        "duo": row, "a_name": a_name,
        "a_pct": round(float(initiator["score"]) * 100) if initiator else 0,
        "a_percentile": initiator["percentile"] if initiator else 0,
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
    return RedirectResponse("/contribute?sent=1", status_code=303)


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


# ═══════════════════════════════════════════════════
# V38 - ADMIN CONSOLE
# ═══════════════════════════════════════════════════

@a.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(req: Request, error: int = 0):
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
    resp = RedirectResponse("/admin/login")
    resp.delete_cookie(chillsauth.ADMIN_COOKIE)
    return resp


@a.get("/admin", response_class=HTMLResponse)
def admin_console(req: Request, tab: str = "overview", sort: str = "created_at", page: int = 0):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")

    total_signups = chillsdb.count_users()
    total_paid = chillsdb.count_paid_users()
    scores = chillsdb.all_scores()
    sends_all = []
    with chillsdb.get_conn() as conn:
        sends_all = conn.execute("SELECT * FROM sends").fetchall()
    responded = [s for s in sends_all if s["status"] in ("watched", "revealed")]
    chills_rate = round(100 * sum(1 for s in responded if s["experienced"]) / len(responded)) if responded else 0
    shared_pct = round(100 * len(sends_all) / total_paid) if total_paid else 0

    per_page = 10
    rows = chillsdb.all_users(order_by=sort, limit=per_page, offset=page * per_page)
    users_view = []
    for u in rows:
        u_sends = chillsdb.sends_for_sender(u["id"])
        users_view.append({
            **dict(u),
            "sent_n": len(u_sends),
            "created_str": datetime.utcfromtimestamp(u["created_at"]).strftime("%b %d"),
        })
    total_pages = max(1, (total_signups + per_page - 1) // per_page)

    return t.TemplateResponse("admin.html", {
        "request": req, "page": "admin", "avatar_letter": "A", "notif_count": 0, "is_admin": True, "has_profile": False,
        "tab": tab, "sort": sort, "page_n": page, "total_pages": total_pages,
        "total_signups": total_signups, "total_paid": total_paid,
        "chills_rate": chills_rate, "shared_pct": shared_pct,
        "users": users_view, "scores_n": len(scores),
    })


@a.get("/admin/export.csv")
def admin_export_csv(req: Request):
    if not chillsauth.is_admin(req):
        return RedirectResponse("/admin/login")
    rows = chillsdb.all_users(limit=100000)
    out = [["signed_up","name","stimulus","score_pct","paid"]]
    for u in rows:
        out.append([
            datetime.utcfromtimestamp(u["created_at"]).isoformat(),
            u["pid"], u["stimulus_name"], round(float(u["score"]) * 100, 1), bool(u["paid"]),
        ])
    csv_text = "\n".join(",".join(f'"{c}"' for c in row) for row in out)
    headers = {"Content-Disposition": "attachment; filename=chillstv_users.csv"}
    return HTMLResponse(csv_text, media_type="text/csv", headers=headers)


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
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if is_new: w.writerow(Hh)
        w.writerow([
            datetime.utcnow().isoformat(), id, email, prolific_id, stimulus_id, url,
            experienced, chills_amount, chills_length, chills_waves,
            description.replace("\r\n","\n").strip()
        ])

    # picked-mode Send Chills threads its token explicitly through the form;
    # algo-mode has no send_token here, it closes via the recipient's own
    # pending_send_token instead (set when they finished the questionnaire).
    if send_token and chillsdb.get_send_by_token(send_token):
        chillsdb.record_send_response(send_token, experienced == "yes",
                                       intensity=chills_amount, recipient_name=id)
    else:
        user = chillsauth.get_current_user(req)
        if user and user["pending_send_token"] and user["stimulus_id"] == stimulus_id:
            chillsdb.record_send_response(user["pending_send_token"], experienced == "yes",
                                           intensity=chills_amount, recipient_name=id)
            chillsdb.set_pending_send_token(user["token"], "")

    return t.TemplateResponse("done.html", {"request": req, "id": id, "email": email})


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

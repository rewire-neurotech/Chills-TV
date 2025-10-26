from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import csv, hashlib, json, os, re, unicodedata
from datetime import datetime
import joblib, numpy as np, onnxruntime as rt, pandas as pd
import sklearn
from typing import Any, Dict, List, Tuple, Optional

# ---------- app + paths ----------
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
    "chosen_head_idx": None,     # <- debug: which ONNX head was used
    "chosen_head_name": None,    # <- debug: which ONNX head was used
}
LP = os.path.join(b, "debug_probs.csv")
FW = {"pairs": [], "missing": [], "raw_answers": {}, "built_vector": []}

# ---------- fixed 40 stimuli (order must match training) ----------
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

# ---------- utils ----------
def r1(p):
    try:
        return pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.read_csv(p, encoding="latin-1")

def nm(x): return re.sub(r"[^0-9a-zA-Z_]+", "_", str(x).strip())

def cq(x):
    x = (x or "").replace("â€™", "’").replace("â€œ", "“").replace("â€�", "”")
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
    s = s.replace("&"," and ").replace("’","'").replace("‘","'").replace("“","\"").replace("”","\"")
    s = re.sub(r"\(audio\)","", s)
    s = re.sub(r"[^a-z0-9]+"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

# ---------- questionnaire ----------
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

# ---------- model + pre ----------
with open(ff, "r", encoding="utf-8") as f:
    F = json.load(f)["features"]

pre = joblib.load(pf)
for c, d in [("_name_to_fitted_passthrough", {}), ("_remainder", "drop")]:
    if not hasattr(pre, c):
        try: setattr(pre, c, d)
        except: pass

sess = rt.InferenceSession(mf, providers=["CPUExecutionProvider"])
inn = sess.get_inputs()[0].name

# ---------- stimuli table ----------
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

# ---------- CSV matching ----------
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
for i in range(len(G)):
    c = canon(G.iloc[i].get(CSV_NAME, ""))
    if c:
        CSV_IDX[c] = i
        CSV_CANON_ROWS.append((i, c))

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

# ---------- feature plumbing ----------
with open(ff, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)["features"]

def ag(x):
    x = str(x or "").strip()
    try: return float(x)
    except:
        y = re.sub(r"\s+","",x)
        m = re.match(r"^(\d+)[\-–](\d+)$", y)
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

# ---------- ONNX inference helpers ----------
CHILLS_HEAD_ENV = os.getenv("REWIRE_CHILLS_HEAD_INDEX")  # optional override
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

    # --- FORCE "probabilities" head if present ---
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
        # Case A: probabilities is a sequence of per-head tensors (e.g., 5 heads)
        if isinstance(y, (list, tuple)):
            try:
                head0 = y[0]  # engineer: probs_0 is CHILLS
            except Exception as ex:
                raise RuntimeError(f"'probabilities' is a sequence but empty/invalid: type={type(y)}") from ex
            arr = np.asarray(head0)
            if arr.ndim == 2 and arr.shape[0] == len(STIM) and arr.shape[1] >= 2:
                p = arr[:, 1].astype(np.float32)  # class-1 = Chills_bin
            elif arr.ndim == 1 and arr.shape[0] == len(STIM):
                p = arr.astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected shape for CHILLS head0: {arr.shape}; expected (40,2) or (40,).")
        else:
            # Case B: structured blob / ndarray with embedded probs
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

    # --- If no 'probabilities', pick CHILLS head by hint/index
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

    # jitter then rank
    eps = (np.arange(len(STIM)) * 1e-9).astype(np.float32)
    p = p + eps
    if np.max(p) - np.min(p) < 1e-6:
        z = (p - np.mean(p)) / (np.std(p) + 1e-9)
        idx = np.argsort(-z)[:k]
    else:
        idx = np.argsort(-p)[:k]

    # record debug
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

    # outputs
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
            "name": n1, "desc": d0, "dur": du, "cap": c0
        })
    return o

# ---------- tie-handling helpers (ONE video output) ----------
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

# ---------- routes ----------
@a.get("/", response_class=HTMLResponse)
def index(req: Request):
    q2 = [x for x in Q if x["k"] != "Age"]
    return t.TemplateResponse("index.html", {"request": req, "DEMO": qdemo(), "QS": q2})

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

@a.post("/start", response_class=HTMLResponse)
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

        d = {"id": pid, "score": best["score"], "stimulus_id": best["stimulus_id"], "url": best["url"],
             "name": best["name"], "desc": best["desc"], "dur": best["dur"], "cap": best["cap"]}
        return t.TemplateResponse("stimulus.html", {"request": req, "D": d})
    except Exception as ex:
        E["msg"] = str(ex); E["when"] = datetime.utcnow().isoformat()
        return HTMLResponse(f"<pre>Internal error during /start\n\n{E['msg']}\n\nCheck /_debug/feature_wire and /_debug/stim_match</pre>", status_code=500)

@a.get("/feedback", response_class=HTMLResponse)
def feedback(req: Request, id: str, stimulus_id: str, url: str = "", score: float = 0.0, stimulus_name: str = ""):
    return t.TemplateResponse("feedback.html", {
        "request": req, "id": id, "stimulus_id": stimulus_id, "url": url,
        "score": score, "stimulus_name": stimulus_name, "class_idx": -1
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
    email: str = Form("")
):
    p = os.path.join(b, "logs.csv")
    Hh = [
        "ts","participant_id","email","stimulus_id","url",
        "experienced","chills_amount_0_10","chills_length_0_6","chills_waves_0_10",
        "description"
    ]
    is_new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if is_new: w.writerow(Hh)
        w.writerow([
            datetime.utcnow().isoformat(), id, email, stimulus_id, url,
            experienced, chills_amount, chills_length, chills_waves,
            description.replace("\r\n","\n").strip()
        ])
    return t.TemplateResponse("done.html", {"request": req, "id": id, "email": email})

# ---------- debug ----------
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

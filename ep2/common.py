import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EP2 = os.path.join(ROOT, "ep2")
EP1_DIR = os.path.join(ROOT, "ep1_pln")
PRED_DIR = os.path.join(EP2, "preds")
os.makedirs(PRED_DIR, exist_ok=True)

LABELS = ["c1", "c234", "c5"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
N_FOLDS = 5

URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def load_data():
    train = pd.read_excel(os.path.join(ROOT, "train.xlsx"))
    test = pd.read_excel(os.path.join(ROOT, "test1.xlsx"))
    X = train["resp_text"].fillna("").astype(str)
    y = train["clarity"].map(LABEL2ID).to_numpy()
    X_test = test["resp_text"].fillna("").astype(str)
    return X, y, X_test, test


def load_test2():
    test2 = pd.read_excel(os.path.join(ROOT, "test2.xlsx"))
    return test2["resp_text"].fillna("").astype(str)


def norm_text(s):
    return SPACE_RE.sub(" ", str(s).lower()).strip()


def fold_id():
    """Dobras do EP1 (as do Colab), reconstruídas a partir dos índices salvos com o BERT."""
    path = os.path.join(EP2, "folds.npy")
    if os.path.exists(path):
        return np.load(path)
    fid = np.full(20092, -1, dtype=int)
    for k in range(N_FOLDS):
        fid[np.load(os.path.join(EP1_DIR, f"bert_fold{k}.npz"))["idx"]] = k
    assert (fid >= 0).all()
    np.save(path, fid)
    return fid


def folds():
    fid = fold_id()
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(N_FOLDS)]


def save_pred(name, oof, test, test_folds=None, **extra):
    d = dict(oof=np.asarray(oof, dtype=np.float64), test=np.asarray(test, dtype=np.float64), fold_id=fold_id())
    if test_folds is not None:
        d["test_folds"] = np.asarray(test_folds, dtype=np.float64)
    d.update(extra)
    np.savez(os.path.join(PRED_DIR, f"{name}.npz"), **d)


def load_pred(name):
    z = np.load(os.path.join(PRED_DIR, f"{name}.npz"), allow_pickle=True)
    assert np.array_equal(z["fold_id"], fold_id()), f"{name}: dobras diferentes"
    return {k: z[k] for k in z.files}


def available_preds():
    return sorted(f[:-4] for f in os.listdir(PRED_DIR) if f.endswith(".npz"))


def acc(y, P):
    return float(np.mean(np.asarray(P).argmax(1) == y)) if np.ndim(P) == 2 else float(np.mean(P == y))


def fold_accs(y, pred):
    fid = fold_id()
    return [float(np.mean(pred[fid == k] == y[fid == k])) for k in range(N_FOLDS)]

"""Funções de stacking usadas nos experimentos do EP2."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import folds, load_data, load_pred, load_test2, norm_text

X, y, X_test, _ = load_data()
X_test2 = load_test2()
FOLDS = folds()
GROUPS = np.unique(X.map(norm_text).to_numpy(), return_inverse=True)[1]

EP1_NAMES = ["ep1_baseline_tfidf_lr", "ep1_fase1_tfidf_meta_lr", "ep1_tfidf_svc", "ep1_tfidf_lr", "ep1_char_svc",
             "ep1_tfidf_cnb", "ep1_w2v_lr", "bert_dapt_ep1"]


def repeated_folds(n_rep=5):
    """Dobras extras só para o nível do meta-modelo (as probabilidades de base continuam OOF)."""
    out = [FOLDS]
    for r in range(1, n_rep):
        out.append(list(StratifiedGroupKFold(5, shuffle=True, random_state=1000 + r).split(X, y, GROUPS)))
    return out


REP_FOLDS = repeated_folds()


def get(names, kind="logp", eps=1e-4):
    preds = [load_pred(n) for n in names]
    oof = [p["oof"] for p in preds]
    test = [p["test"] for p in preds]
    if kind == "prob":
        f = lambda A: A
    elif kind == "logp":
        f = lambda A: np.log(np.clip(A, eps, 1))
    else:
        raise ValueError(kind)
    return np.hstack([f(a) for a in oof]), np.hstack([f(a) for a in test])


def lr(C=1.0, scale=False):
    m = LogisticRegression(C=C, max_iter=5000)
    return make_pipeline(StandardScaler(), m) if scale else m


def cv_proba(make_model, Z, y_=None, folds_=None):
    y_ = y if y_ is None else y_
    folds_ = FOLDS if folds_ is None else folds_
    P = np.zeros((len(Z), 3))
    for tr, va in folds_:
        P[va] = make_model().fit(Z[tr], y_[tr]).predict_proba(Z[va])
    return P


def evaluate(make_model, Z, rep=True):
    """Acurácia nas dobras do EP1 e média em 5 repetições de dobras do meta-modelo."""
    P = cv_proba(make_model, Z)
    res = {"acc": float(np.mean(P.argmax(1) == y)), "logloss": float(log_loss(y, P))}
    if rep:
        accs = [np.mean(cv_proba(make_model, Z, folds_=f).argmax(1) == y) for f in REP_FOLDS]
        res["acc_rep"] = float(np.mean(accs))
    return res, P


def paired(pred_a, pred_b):
    """Diferença de acurácia (b - a) e erro padrão pareado."""
    ca, cb = (pred_a == y), (pred_b == y)
    d = cb.astype(float) - ca.astype(float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(((~ca) & cb).sum()), int((ca & ~cb).sum())

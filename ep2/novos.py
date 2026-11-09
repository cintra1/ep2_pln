"""Modelos de base novos (CPU), nas dobras fixas, com previsões do teste por dobra.

Uso: python novos.py knn nbsvm lgb_bow ridge_ord
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from common import acc, folds, load_data, norm_text, save_pred

X, y, X_test, _ = load_data()
FOLDS = folds()
KEY = X.map(norm_text).to_numpy()
KEY_T = X_test.map(norm_text).to_numpy()
PRIOR = np.bincount(y, minlength=3) / len(y)


def run(name, fit_predict):
    """fit_predict(tr_idx, eval_sets) -> lista de probabilidades, uma por conjunto de avaliação."""
    t = time.time()
    oof = np.zeros((len(X), 3))
    test_folds = []
    for tr, va in FOLDS:
        P_va, P_te = fit_predict(tr, [("train", va), ("test", None)])
        oof[va] = P_va
        test_folds.append(P_te)
    (P_full,) = fit_predict(np.arange(len(X)), [("test", None)])
    save_pred(name, oof, P_full, np.stack(test_folds))
    print(f"{name:20s} {acc(y, oof):.4f}  [{time.time() - t:.0f}s]", flush=True)
    return oof


# ---------------------------------------------------------------- kNN sobre TF-IDF (sem rótulo, transdutivo)
_tfidf = None


def tfidf_all():
    global _tfidf
    if _tfidf is None:
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32)
        V = vec.fit_transform(list(X) + list(X_test)).tocsr()
        _tfidf = (V[:len(X)], V[len(X):])
    return _tfidf


def knn_factory(k=30, power=4.0, beta=1.0):
    V, Vt = tfidf_all()

    def fit_predict(tr, sets):
        out = []
        for kind, idx in sets:
            Q = V[idx] if kind == "train" else Vt
            q_keys = KEY[idx] if kind == "train" else KEY_T
            P = np.zeros((Q.shape[0], 3))
            for s in range(0, Q.shape[0], 2000):
                S = (Q[s:s + 2000] @ V[tr].T).toarray()
                same = q_keys[s:s + 2000][:, None] == KEY[tr][None, :]
                S[same] = 0.0
                top = np.argpartition(-S, k, axis=1)[:, :k]
                w = np.take_along_axis(S, top, axis=1) ** power
                lab = y[tr][top]
                for c in range(3):
                    P[s:s + 2000, c] = (w * (lab == c)).sum(1)
            P = (P + beta * PRIOR) / (P.sum(1, keepdims=True) + beta)
            out.append(P)
        return out

    return fit_predict


# ---------------------------------------------------------------- NB-SVM (Wang & Manning, 2012), um contra todos
def nbsvm_factory(C=0.1, alpha=1.0):
    vec_kw = dict(ngram_range=(1, 3), min_df=2, binary=True, max_features=300000, dtype=np.float32)

    def fit_predict(tr, sets):
        vec = CountVectorizer(**vec_kw)
        Xtr = vec.fit_transform(X.iloc[tr])
        scores = []
        mats = [vec.transform(X.iloc[idx]) if kind == "train" else vec.transform(X_test) for kind, idx in sets]
        for c in range(3):
            pos = y[tr] == c
            p = alpha + np.asarray(Xtr[pos].sum(0)).ravel()
            q = alpha + np.asarray(Xtr[~pos].sum(0)).ravel()
            r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
            R = sp.diags(r)
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr @ R, pos)
            scores.append([clf.decision_function(M @ R) for M in mats])
        out = []
        for j in range(len(sets)):
            S = np.stack([scores[c][j] for c in range(3)], 1)
            E = np.exp(S - S.max(1, keepdims=True))
            out.append(E / E.sum(1, keepdims=True))
        return out

    return fit_predict


# ---------------------------------------------------------------- LightGBM sobre bag-of-words binário
def lgb_factory(n_estimators=600):
    import lightgbm as lgb

    def fit_predict(tr, sets):
        vec = CountVectorizer(ngram_range=(1, 2), min_df=10, max_features=40000, binary=True, dtype=np.float32)
        Xtr = vec.fit_transform(X.iloc[tr])
        L = lambda s: np.log1p(np.c_[s.str.len(), s.str.split().str.len()]).astype(np.float32)
        Xtr = sp.hstack([Xtr, L(X.iloc[tr])]).tocsr()
        clf = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                 colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                                 verbose=-1, n_jobs=4)
        clf.fit(Xtr, y[tr])
        out = []
        for kind, idx in sets:
            s = X.iloc[idx] if kind == "train" else X_test
            M = sp.hstack([vec.transform(s), L(s)]).tocsr()
            out.append(clf.predict_proba(M))
        return out

    return fit_predict


# ---------------------------------------------------------------- regressão ordinal (ridge em 0/1/2)
def ridge_factory(alpha=3.0):
    def fit_predict(tr, sets):
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, dtype=np.float32)
        Xtr = vec.fit_transform(X.iloc[tr])
        reg = Ridge(alpha=alpha).fit(Xtr, y[tr].astype(float))
        z_tr = reg.predict(Xtr)
        # converte o escore contínuo em probabilidades com uma LR de 1 variável (ajustada no próprio treino)
        cal = LogisticRegression(max_iter=1000).fit(np.c_[z_tr, z_tr ** 2], y[tr])
        out = []
        for kind, idx in sets:
            z = reg.predict(vec.transform(X.iloc[idx] if kind == "train" else X_test))
            out.append(cal.predict_proba(np.c_[z, z ** 2]))
        return out

    return fit_predict


if __name__ == "__main__":
    todo = sys.argv[1:]
    if "knn" in todo:
        for k, pw in [(30, 4.0), (100, 4.0), (100, 1.0)]:
            run(f"knn_k{k}_p{int(pw)}", knn_factory(k, pw))
    if "nbsvm" in todo:
        run("nbsvm", nbsvm_factory())
    if "lgb_bow" in todo:
        run("lgb_bow", lgb_factory())
    if "ridge_ord" in todo:
        run("ridge_ord", ridge_factory())

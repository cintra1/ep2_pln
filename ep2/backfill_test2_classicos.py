"""Adiciona previsoes de teste2 aos modelos classicos do EP1 (ep1_*) e ao nbsvm, que foram
cacheados so com teste1 (teste2 nao existia no EP1). Reconstroi os mesmos pipelines usados no
notebook do EP1 (desenvolvimento/ep1_pln do projeto antigo), re-treina em (X, y) completo (igual
ao que gerou o campo "test" original) e preve em test2, sem tocar oof/test ja cacheados.

Uso: python backfill_test2_classicos.py
"""
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from common import PRED_DIR, acc, load_data, load_pred, load_test2

SEED = 42
X, y, X_test, _ = load_data()
X_test2 = load_test2()

URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def preprocess(text):
    text = str(text).lower()
    text = URL_RE.sub(" __url__ ", text)
    text = EMAIL_RE.sub(" __email__ ", text)
    text = NUM_RE.sub(" __num__ ", text)
    return SPACE_RE.sub(" ", text).strip()


def meta_features(texts):
    s = pd.Series(list(texts), dtype=str)
    return np.log1p(np.c_[
        s.str.len(),
        s.str.split().str.len(),
        s.str.count(URL_RE.pattern),
        s.str.count(r"\s{2,}"),
        s.str.count(NUM_RE.pattern),
    ])


def make_official_baseline():
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def make_phase1_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=preprocess, sublinear_tf=True, ngram_range=(1, 2), min_df=5)),
            ("meta", make_pipeline(FunctionTransformer(meta_features), StandardScaler())),
        ])),
        ("clf", LogisticRegression(C=0.3, max_iter=3000)),
    ])


analisador = TfidfVectorizer().build_analyzer()


class DocVectorW2V(BaseEstimator, TransformerMixin):
    def __init__(self, w2v=None, ponderacao="tfidf", k=None):
        self.w2v = w2v
        self.ponderacao = ponderacao
        self.k = k

    def fit(self, X, y=None):
        wv = getattr(self.w2v, "wv", self.w2v)
        vocab = list(wv.index_to_key)
        if self.ponderacao == "media":
            self.vec_ = CountVectorizer(analyzer=analisador, vocabulary=vocab)
        else:
            self.vec_ = TfidfVectorizer(analyzer=analisador, vocabulary=vocab, sublinear_tf=True)
        W = self.vec_.fit_transform(X)
        self.E_ = np.asarray(wv.vectors, dtype=np.float64)
        self.mask_ = None
        if self.k is not None:
            self.mask_ = SelectKBest(chi2, k=min(self.k, W.shape[1])).fit(W, y).get_support()
        return self

    def transform(self, X):
        W = self.vec_.transform(X)
        if self.mask_ is not None:
            W = W.multiply(self.mask_[None, :]).tocsr()
        soma = np.asarray(W.sum(axis=1)).ravel()
        soma[soma == 0] = 1.0
        return (W @ self.E_) / soma[:, None]


def svc_calibrado(C=0.1):
    return CalibratedClassifierCV(LinearSVC(C=C), method="sigmoid", cv=5)


def tfidf_palavras():
    return TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)


def modelo_tfidf_lr():
    return Pipeline([("tfidf", tfidf_palavras()),
                      ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))])


def modelo_tfidf_svc():
    return Pipeline([("tfidf", tfidf_palavras()), ("clf", svc_calibrado(0.1))])


def modelo_tfidf_cnb():
    return Pipeline([("tfidf", tfidf_palavras()), ("clf", ComplementNB(alpha=1.0))])


def modelo_char_svc():
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3,
                                                max_features=200000, dtype=np.float32)),
                      ("clf", svc_calibrado(0.1))])


def modelo_w2v_lr(w2v):
    return Pipeline([("doc", DocVectorW2V(w2v, ponderacao="tfidf", k=9000)),
                      ("scaler", StandardScaler()),
                      ("clf", LogisticRegression(C=0.1, max_iter=3000))])


CLASSIC_MODELS = {
    "baseline_tfidf_lr": make_official_baseline,
    "fase1_tfidf_meta_lr": make_phase1_model,
    "tfidf_svc": modelo_tfidf_svc,
    "tfidf_lr": modelo_tfidf_lr,
    "char_svc": modelo_char_svc,
    "tfidf_cnb": modelo_tfidf_cnb,
}


def add_test2(name, test2_probs):
    z = dict(load_pred(name))
    if "test2" in z and np.allclose(z["test2"], test2_probs):
        print(f"{name:25s} test2 ja presente e igual, nada a fazer.")
        return
    z["test2"] = np.asarray(test2_probs, dtype=np.float64)
    np.savez(os.path.join(PRED_DIR, f"{name}.npz"), **z)
    sanity = acc(y, z["oof"])
    print(f"{name:25s} test2 adicionado (OOF do cache inalterado: {sanity:.4f})")


def main():
    for name, builder in CLASSIC_MODELS.items():
        model = builder().fit(X, y)
        assert list(model.classes_) == [0, 1, 2]
        P2 = model.predict_proba(X_test2)
        add_test2(f"ep1_{name}", P2)

    print("Treinando Word2Vec (igual ao EP1, seed=42) ...")
    w2v = Word2Vec([analisador(s) for s in pd.concat([X, X_test])], vector_size=300, window=5, min_count=2,
                    sg=1, seed=SEED, epochs=10, workers=4)
    model = modelo_w2v_lr(w2v).fit(X, y)
    add_test2("ep1_w2v_lr", model.predict_proba(X_test2))

    print("\nNB-SVM ...")
    vec_kw = dict(ngram_range=(1, 3), min_df=2, binary=True, max_features=300000, dtype=np.float32)
    vec = CountVectorizer(**vec_kw)
    Xtr = vec.fit_transform(X)
    M2 = vec.transform(X_test2)
    scores2 = []
    for c in range(3):
        pos = y == c
        p = 1.0 + np.asarray(Xtr[pos].sum(0)).ravel()
        q = 1.0 + np.asarray(Xtr[~pos].sum(0)).ravel()
        r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        R = sp.diags(r)
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(Xtr @ R, pos)
        scores2.append(clf.decision_function(M2 @ R))
    S = np.stack(scores2, 1)
    E = np.exp(S - S.max(1, keepdims=True))
    add_test2("nbsvm", E / E.sum(1, keepdims=True))


if __name__ == "__main__":
    main()

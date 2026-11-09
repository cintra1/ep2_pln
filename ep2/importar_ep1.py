"""Converte as probabilidades do EP1 (pasta ep1_pln) para ep2/preds/<modelo>.npz e reproduz o stacking."""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from common import EP1_DIR, N_FOLDS, acc, fold_id, folds, load_data, save_pred

X, y, X_test, _ = load_data()
fid = fold_id()

oof = np.zeros((len(X), 3))
test_folds = np.zeros((N_FOLDS, len(X_test), 3))
epoch_oof = np.zeros((3, len(X), 3))
for k in range(N_FOLDS):
    z = np.load(os.path.join(EP1_DIR, f"bert_fold{k}.npz"))
    oof[z["idx"]] = z["oof"]
    epoch_oof[:, z["idx"]] = z["epoch_oof"]
    test_folds[k] = z["test"]
save_pred("bert_dapt_ep1", oof, test_folds.mean(0), test_folds)
for e in range(3):
    print(f"BERT época {e + 1}: {acc(y, epoch_oof[e]):.4f}")

z = np.load(os.path.join(EP1_DIR, "modelos_classicos.npz"))
print("fold_id do cache clássico igual ao do BERT:", np.array_equal(z["fold_id"], fid))
NAMES = ["baseline_tfidf_lr", "fase1_tfidf_meta_lr", "tfidf_svc", "tfidf_lr", "char_svc", "tfidf_cnb", "w2v_lr"]
for n in NAMES:
    save_pred(f"ep1_{n}", z[f"{n}_oof"], z[f"{n}_test"])

P = {n: z[f"{n}_oof"] for n in NAMES}
P["bert"] = oof
T = {n: z[f"{n}_test"] for n in NAMES}
T["bert"] = test_folds.mean(0)
for n in P:
    print(f"{n:22s} {acc(y, P[n]):.4f}")

Z = np.hstack([P[n] for n in P])
Zt = np.hstack([T[n] for n in T])
meta = LogisticRegression(C=1.0, max_iter=2000)
pred = cross_val_predict(meta, Z, y, cv=folds())
print(f"\nStacking EP1 reproduzido: {np.mean(pred == y):.4f}")

import pandas as pd
from common import LABELS, ROOT
meta.fit(Z, y)
entregue = pd.read_excel(os.path.join(ROOT, "ep1_test1_rotulado.xlsx"))["clarity"].to_numpy()
mine = np.array(LABELS)[meta.predict(Zt)]
print("concordância com a planilha entregue no EP1:", np.mean(mine == entregue))

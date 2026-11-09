"""Importa os resultados do Colab colocados em ep2/gpu_runs.

- <PRESET>.npz (junção feita pelo notebook) ou <PRESET>_fold<k>.npz (as 5 dobras, de uma ou várias contas)
  viram ep2/preds/gpu_<PRESET>.npz (época final) e gpu_<PRESET>_epoca<e>.npz (épocas anteriores).
- emb_<nome>.npy (notebook de embeddings) vira ep2/preds/emb_<nome>.npz, com uma regressão logística nas dobras.

Cada modelo carrega, além do OOF/teste1 (campos "oof"/"test"/"test_folds", usados pelo stacking do EP1), as
probabilidades do teste2 (897 textos, entrega oficial) em "test2"/"test2_folds" — ainda não consumidas pelo
stack_final.py, que por enquanto só prevê o teste1.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common import EP2, ROOT, acc, fold_id, folds, load_data, save_pred

GPU_DIR = os.path.join(EP2, "gpu_runs")
os.makedirs(GPU_DIR, exist_ok=True)
X, y, X_test, _ = load_data()
X_test2 = pd.read_excel(os.path.join(ROOT, "test2.xlsx"))["resp_text"].fillna("").astype(str).tolist()
FID = fold_id()


def import_finetune():
    runs = {}
    for path in glob.glob(os.path.join(GPU_DIR, "*.npz")):
        m = re.match(r"(.+)_fold(\d)\.npz$", os.path.basename(path))
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = path
        else:
            runs.setdefault(os.path.basename(path)[:-4], {})["merged"] = path
    for run, files in sorted(runs.items()):
        if "merged" in files:
            z = np.load(files["merged"])
            assert np.array_equal(z["fold_id"], FID), f"{run}: dobras diferentes"
            epoch_oof = z["epoch_oof"]
            epoch_test1_folds, epoch_test2_folds = z["epoch_test1_folds"], z["epoch_test2_folds"]
        elif all(k in files for k in range(5)):
            zs = [np.load(files[k]) for k in range(5)]
            n_ep = min(z["epoch_oof"].shape[0] for z in zs)
            epoch_oof = np.zeros((n_ep, len(X), 3))
            for k, z in enumerate(zs):
                assert np.array_equal(np.sort(z["idx"]), np.where(FID == k)[0]), f"{run}: dobra {k} diferente"
                epoch_oof[:, z["idx"]] = z["epoch_oof"][:n_ep]
            epoch_test1_folds = np.stack([z["epoch_test1"][:n_ep] for z in zs], axis=1)
            epoch_test2_folds = np.stack([z["epoch_test2"][:n_ep] for z in zs], axis=1)
        else:
            print(f"{run}: dobras incompletas {sorted(k for k in files if k != 'merged')}, ignorado")
            continue
        n_ep = epoch_oof.shape[0]
        for e in range(n_ep):
            name = f"gpu_{run}" if e == n_ep - 1 else f"gpu_{run}_epoca{e + 1}"
            save_pred(name, epoch_oof[e], epoch_test1_folds[e].mean(0), epoch_test1_folds[e],
                      test2=epoch_test2_folds[e].mean(0), test2_folds=epoch_test2_folds[e])
            print(f"{name:40s} OOF {acc(y, epoch_oof[e]):.4f}")


def import_embeddings(Cs=(0.03, 0.1, 0.3, 1.0, 3.0)):
    n_tr, n_te1, n_te2 = len(X), len(X_test), len(X_test2)
    for path in sorted(glob.glob(os.path.join(GPU_DIR, "emb_*.npy"))):
        name = os.path.basename(path)[:-4]
        E = np.load(path).astype(np.float32)
        assert E.shape[0] == n_tr + n_te1 + n_te2, f"{name}: {E.shape}"
        E = E - E[:n_tr].mean(0)
        E /= E[:n_tr].std(0) + 1e-6
        Etr, Ete1, Ete2 = E[:n_tr], E[n_tr:n_tr + n_te1], E[n_tr + n_te1:]
        best = None
        for C in Cs:
            oof = np.zeros((n_tr, 3))
            test1_folds = []
            for tr, va in folds():
                clf = LogisticRegression(C=C, max_iter=3000).fit(Etr[tr], y[tr])
                oof[va] = clf.predict_proba(Etr[va])
                test1_folds.append(clf.predict_proba(Ete1))
            ll = log_loss(y, oof)
            print(f"  {name} C={C}: acc {acc(y, oof):.4f} logloss {ll:.4f}")
            if best is None or ll < best[0]:
                best = (ll, C, oof, np.stack(test1_folds))
        ll, C, oof, test1_folds = best
        full = LogisticRegression(C=C, max_iter=3000).fit(Etr, y)
        save_pred(name, oof, full.predict_proba(Ete1), test1_folds, C=C, test2=full.predict_proba(Ete2))
        print(f"{name:40s} OOF {acc(y, oof):.4f} (C={C})")


if __name__ == "__main__":
    import_finetune()
    import_embeddings()

"""Textos idênticos (após minúsculas e espaços normalizados) entre treino e teste.

No OOF, as cópias de um texto nunca são vistas pelo modelo (dobras agrupadas). No teste, um texto com cópia no
treino teria sido "decorado" pelos modelos treinados no treino todo. Para manter treino e teste na mesma situação:
- previsões de base do teste para textos com cópia vêm dos modelos das dobras que contêm essas cópias (que não as
  viram no treino), e não do modelo do treino completo;
- os rótulos das cópias entram depois, como contagens (leave-one-out no treino).
"""
import numpy as np
import pandas as pd

from common import fold_id, norm_text
from stack_lib import X, X_test, X_test2, y

KEY = X.map(norm_text).to_numpy()
KEY_T = X_test.map(norm_text).to_numpy()
KEY_T2 = X_test2.map(norm_text).to_numpy()
FID = fold_id()

_counts = pd.crosstab(KEY, y).reindex(columns=[0, 1, 2], fill_value=0)
C_LOO = _counts.reindex(KEY).to_numpy().astype(float) - np.eye(3)[y]
C_TEST = _counts.reindex(KEY_T).fillna(0).to_numpy().astype(float)
C_TEST2 = _counts.reindex(KEY_T2).fillna(0).to_numpy().astype(float)

# dobras em que estão as cópias de cada texto do teste
_folds_of_key = pd.Series(FID).groupby(KEY).agg(lambda s: sorted(set(s)))
TEST_DUP_FOLDS = [(_folds_of_key.get(k, [])) for k in KEY_T]
HAS_TEST_DUP = np.array([len(f) > 0 for f in TEST_DUP_FOLDS])
TEST2_DUP_FOLDS = [(_folds_of_key.get(k, [])) for k in KEY_T2]
HAS_TEST2_DUP = np.array([len(f) > 0 for f in TEST2_DUP_FOLDS])


def unseen_test(pred):
    """Probabilidades do teste em que os textos com cópia usam só os modelos que não viram a cópia."""
    T = pred["test"].copy()
    if "test_folds" not in pred:
        raise ValueError("modelo sem previsões do teste por dobra")
    for i, fl in enumerate(TEST_DUP_FOLDS):
        if fl:
            T[i] = pred["test_folds"][fl, i].mean(0)
    return T


def unseen_test2(pred):
    """Mesma ideia de unseen_test, para o teste2."""
    T = pred["test2"].copy()
    if "test2_folds" not in pred:
        raise ValueError("modelo sem previsões do teste2 por dobra")
    for i, fl in enumerate(TEST2_DUP_FOLDS):
        if fl:
            T[i] = pred["test2_folds"][fl, i].mean(0)
    return T


def dirichlet(P, counts, alpha):
    n = counts.sum(1, keepdims=True)
    return (alpha * P + counts) / (alpha + n)

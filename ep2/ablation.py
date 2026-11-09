"""Ablacao do stacking: o quanto cada modelo contribui para o OOF final (nunca mexe nas planilhas
de entrega, so calcula acuracia). Reusa a mesma logica de stack_final.py (grade de C do meta-modelo +
suavizacao de Dirichlet pelas copias), sem escrever nada em disco.

Uso: python ablation.py
"""
import numpy as np

from stack_final import ALPHAS, C_GRID, all_models, build, choose_alpha
from dup import C_LOO, dirichlet
from stack_lib import FOLDS, cv_proba, lr, y

NAMES = all_models()


def oof_accuracy(names):
    _, Z, _ = build(names)

    def score(C):
        P = cv_proba(lambda: lr(C), Z)
        pred = np.empty(len(y), dtype=int)
        for tr, va in FOLDS:
            pred[va] = dirichlet(P[va], C_LOO[va], choose_alpha(P, tr)).argmax(1)
        return float(np.mean(pred == y))

    return max(score(C) for C in C_GRID)


print(f"Modelos no ensemble final: {len(NAMES)}")
full = oof_accuracy(NAMES)
print(f"\nTodos os {len(NAMES)} modelos: {full:.4f}  (referencia)\n")

print("--- tirando um modelo de cada vez (quanto PIOR fica sem ele = contribuicao) ---")
deltas = {}
for n in NAMES:
    sub = [m for m in NAMES if m != n]
    acc = oof_accuracy(sub)
    deltas[n] = full - acc
    sinal = "ajuda" if acc < full else ("neutro" if acc == full else "atrapalha")
    print(f"  sem {n:32s} -> {acc:.4f}  (full - sem_ele = {full - acc:+.4f}, {sinal})")

print("\n--- ordenado por contribuicao (maior queda ao remover = mais importante) ---")
for n, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
    print(f"  {n:32s} {d:+.4f}")

print("\n--- so os 8 modelos do EP1 + nbsvm (sem nenhum dos 5 novos) ---")
base9 = [n for n in NAMES if not n.startswith("gpu_")]
print(f"  {len(base9)} modelos: {oof_accuracy(base9):.4f}")

print("\n--- so os 5 modelos novos (gpu_*) ---")
novos5 = [n for n in NAMES if n.startswith("gpu_")]
print(f"  {len(novos5)} modelos: {oof_accuracy(novos5):.4f}")

"""Selecao greedy para tras: remove, um de cada vez, o modelo cuja remocao mais melhora o OOF,
repete ate que remover qualquer um so piore. Nao mexe em planilhas de entrega.
"""
from ablation import oof_accuracy
from stack_final import all_models

current = all_models()
best = oof_accuracy(current)
print(f"Inicio: {len(current)} modelos, OOF = {best:.4f}\n")

while True:
    candidates = {}
    for n in current:
        sub = [m for m in current if m != n]
        candidates[n] = oof_accuracy(sub)
    worst_to_keep = max(candidates, key=candidates.get)
    if candidates[worst_to_keep] > best:
        print(f"Remove {worst_to_keep:32s} -> OOF sobe de {best:.4f} para {candidates[worst_to_keep]:.4f}")
        current.remove(worst_to_keep)
        best = candidates[worst_to_keep]
    else:
        print(f"Nenhuma remocao melhora mais (melhor tentativa seria tirar {worst_to_keep} -> "
              f"{candidates[worst_to_keep]:.4f}, pior que {best:.4f}). Parando.")
        break

print(f"\nResultado final: {len(current)} modelos, OOF = {best:.4f}")
print(sorted(current))

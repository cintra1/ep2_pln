"""Stacking do EP2 e planilha do teste.

1. Meta-modelo: regressão logística multinomial sobre as probabilidades OOF dos modelos de base.
2. Cópias idênticas: a previsão do meta-modelo é combinada com as contagens de rótulo das cópias do texto no
   treino, P = (alpha * P_meta + contagens) / (alpha + n). No treino as contagens excluem a própria linha.
   No teste, as probabilidades de base dos textos com cópia vêm dos modelos que não viram essas cópias.

Uso: python stack_final.py [modelo1 modelo2 ...]   (sem argumentos: todos os modelos em ep2/preds)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from common import EP2, LABELS, ROOT, available_preds, fold_accs, load_pred
from dup import C_LOO, C_TEST, C_TEST2, HAS_TEST_DUP, HAS_TEST2_DUP, dirichlet, unseen_test, unseen_test2
from stack_lib import EP1_NAMES, FOLDS, REP_FOLDS, X_test, X_test2, cv_proba, lr, paired, y

C_GRID = [0.03, 0.1, 0.3, 1.0]
ALPHAS = [1, 2, 3, 4, 6, 8, 12, 16]
# OOF do EP1 (ep1_*) ganhou dos clássicos refeitos (cls_*). gpu_/emb_ entram quando existirem.
CORE = list(EP1_NAMES) + ["nbsvm"]
EXCLUDE_SUB = ("knn_", "hash_", "estilo_", "_epoca", "cls_")

# Modelo final: os 13 modelos (os 7 clássicos do EP1 + nbsvm + 5 modelos neurais novos, sem o
# bert_dapt_ep1 original por não ter teste2). Fixo por nome (não usa all_models() direto) para que um
# novo modelo em gpu_runs/ não entre automaticamente no ensemble oficial sem revisão.
# - Uma eliminação reversa gulosa (ablation.py / greedy_backward.py) achou um subconjunto de 9 modelos
#   com OOF maior (0,4800 x 0,4782), mas que perde a significância no McNemar contra o EP1 no teste1
#   (p=0,233 x p=0,047) - amostra pequena (900 textos) e risco de sobreajuste da própria busca.
# - Testamos acrescentar o albertina100m (DeBERTa, arquitetura nova): OOF cai para 0,4780 e o McNemar
#   também perde a significância (p=0,320) - não entra no ensemble final.
FINAL_MODELS = ["ep1_baseline_tfidf_lr", "ep1_fase1_tfidf_meta_lr", "ep1_tfidf_svc", "ep1_tfidf_lr",
                 "ep1_char_svc", "ep1_tfidf_cnb", "ep1_w2v_lr", "nbsvm", "gpu_bertikal",
                 "gpu_bertimbau_base_dapt_s7", "gpu_bertimbau_large", "gpu_govbert_br", "gpu_legalbertimbau"]


def all_models():
    """Todos os modelos disponíveis com teste2 (o ensemble final de 13 modelos)."""
    have = set(available_preds())
    core = [n for n in CORE if n in have]
    extra = [n for n in available_preds()
             if n not in core and n.startswith(("gpu_", "emb_")) and "_epoca" not in n]
    names = core + extra
    missing_test2 = [n for n in names if "test2" not in load_pred(n)]
    if missing_test2:
        print(f"Aviso: sem teste2, excluidos do ensemble: {missing_test2}")
        names = [n for n in names if n not in missing_test2]
    return names


def default_models():
    if FINAL_MODELS is None:
        return all_models()
    have = set(available_preds())
    return [n for n in FINAL_MODELS if n in have]


def build(names):
    preds = {n: load_pred(n) for n in names}
    Z = np.hstack([preds[n]["oof"] for n in names])
    Zt = np.hstack([unseen_test(preds[n]) if "test_folds" in preds[n] else preds[n]["test"] for n in names])
    return preds, Z, Zt


def choose_alpha(P, rows):
    accs = [np.mean(dirichlet(P[rows], C_LOO[rows], a).argmax(1) == y[rows]) for a in ALPHAS]
    return ALPHAS[int(np.argmax(accs))]


def mcnemar(correct_a, correct_b):
    """Teste de McNemar pareado: discorda a->certo/b->errado (b_) vs a->errado/b->certo (c_).

    Com poucas discordâncias (n < 25), usa o teste exato (binomial); senão, a aproximação qui-quadrado
    com correção de continuidade.
    """
    b_ = int((correct_a & ~correct_b).sum())
    c_ = int((~correct_a & correct_b).sum())
    n = b_ + c_
    if n == 0:
        return dict(b=b_, c=c_, stat=None, p=1.0, metodo="sem discordâncias")
    if n < 25:
        p = binomtest(min(b_, c_), n, 0.5, alternative="two-sided").pvalue
        return dict(b=b_, c=c_, stat=None, p=float(p), metodo="exato (binomial)")
    stat = (abs(b_ - c_) - 1) ** 2 / n
    return dict(b=b_, c=c_, stat=float(stat), p=float(chi2.sf(stat, df=1)),
                metodo="qui-quadrado com correção de continuidade")


def main(names):
    preds, Z, Zt = build(names)
    print("Modelos de base:")
    for n in names:
        print(f"  {n:40s} OOF {np.mean(preds[n]['oof'].argmax(1) == y):.4f}")

    def score_with_dups(C):
        P = cv_proba(lambda: lr(C), Z)
        pred = np.empty(len(y), dtype=int)
        for tr, va in FOLDS:
            pred[va] = dirichlet(P[va], C_LOO[va], choose_alpha(P, tr)).argmax(1)
        rep = np.mean([np.mean(cv_proba(lambda: lr(C), Z, folds_=f).argmax(1) == y) for f in REP_FOLDS])
        return float(np.mean(pred == y)), float(rep), P, pred

    grid = {C: score_with_dups(C) for C in C_GRID}
    C = max(grid, key=lambda c: (grid[c][0], grid[c][1]))
    print("\nMeta-modelo (OOF + cópias | rep sem cópias):",
          {c: (round(grid[c][0], 4), round(grid[c][1], 4)) for c in C_GRID}, "-> C =", C)
    _, _, P, pred = grid[C]
    pred_meta = P.argmax(1)
    alpha = choose_alpha(P, np.arange(len(y)))

    Zr, _ = np.hstack([load_pred(n)["oof"] for n in EP1_NAMES]), None
    pred_ep1 = cv_proba(lambda: lr(1.0), Zr).argmax(1)
    print(f"\nEP1 (referência)                         {np.mean(pred_ep1 == y):.4f}")
    print(f"Stacking EP2 (sem cópias)                {np.mean(pred_meta == y):.4f}")
    print(f"Stacking EP2 + cópias (alpha={alpha:<3})         {np.mean(pred == y):.4f}  "
          f"| dobras {[round(a, 4) for a in fold_accs(y, pred)]}")
    d, se, win, lose = paired(pred_ep1, pred)
    print(f"Ganho sobre o EP1: {d * 100:+.2f} pontos (erro padrão pareado {se * 100:.2f}; {win} acertos novos, "
          f"{lose} perdidos)")

    meta = lr(C).fit(Z, y)
    P_test = dirichlet(meta.predict_proba(Zt), C_TEST, alpha)
    labels = np.array(LABELS)[P_test.argmax(1)]
    test = pd.read_excel(os.path.join(ROOT, "test1.xlsx"))
    sub = test.copy()
    sub["clarity"] = labels
    out = os.path.join(EP2, "ep2_test1_rotulado.xlsx")
    sub.to_excel(out, index=False)

    check = pd.read_excel(out)
    assert check.shape == test.shape == (900, 2)
    assert list(check.columns) == ["resp_text", "clarity"]
    assert check["resp_text"].fillna("").astype(str).equals(X_test)
    assert set(check["clarity"]) <= set(LABELS) and check["clarity"].notna().all()
    ep1 = pd.read_excel(os.path.join(ROOT, "ep1_test1_rotulado.xlsx"))["clarity"].to_numpy()
    print(f"\nPlanilha: {out}")
    print("Distribuição:", check["clarity"].value_counts().to_dict(),
          f"| concordância com a do EP1: {np.mean(labels == ep1):.1%}",
          f"| textos com cópia no treino: {HAS_TEST_DUP.sum()}")

    # ---- teste2: entrega oficial (897 rótulos) ----
    missing_test2 = [n for n in names if "test2" not in preds[n]]
    if missing_test2:
        print(f"\nTeste2 PULADO: sem previsões de teste2 para {missing_test2}.")
        print("Rode importar_gpu.py depois do Colab (o notebook já gera teste1 e teste2 juntos) e rode de novo.")
    else:
        Zt2 = np.hstack([unseen_test2(preds[n]) if "test2_folds" in preds[n] else preds[n]["test2"]
                         for n in names])
        P_test2 = dirichlet(meta.predict_proba(Zt2), C_TEST2, alpha)
        labels2 = np.array(LABELS)[P_test2.argmax(1)]
        test2 = pd.read_excel(os.path.join(ROOT, "test2.xlsx"))
        sub2 = test2.copy()
        sub2["clarity"] = labels2
        out2_dir = os.path.join(ROOT, "entrega", "teste2_resultado")
        os.makedirs(out2_dir, exist_ok=True)
        out2 = os.path.join(out2_dir, "ep2_test2_rotulado.xlsx")
        sub2.to_excel(out2, index=False)

        check2 = pd.read_excel(out2)
        assert check2.shape == test2.shape == (897, 2)
        assert list(check2.columns) == ["resp_text", "clarity"]
        assert check2["resp_text"].fillna("").astype(str).equals(X_test2)
        assert set(check2["clarity"]) <= set(LABELS) and check2["clarity"].notna().all()
        print(f"\nPlanilha teste2 (entrega oficial): {out2}")
        print("Distribuição teste2:", check2["clarity"].value_counts().to_dict(),
              f"| textos com cópia no treino: {HAS_TEST2_DUP.sum()}")

    # ---- McNemar EP1 x EP2 no teste1, com o ouro (só para o relatório; nunca usar o teste2 aqui) ----
    gold = pd.read_excel(os.path.join(ROOT, "test1_ouro_mcnemar.xlsx"))
    assert gold["resp_text"].fillna("").astype(str).equals(X_test), "test1_ouro_mcnemar.xlsx fora de ordem"
    gold_y = gold["clarity"].to_numpy()
    correct_ep1 = ep1 == gold_y
    correct_ep2 = labels == gold_y
    m = mcnemar(correct_ep1, correct_ep2)
    print(f"\nMcNemar EP1 x EP2 no teste1 (ouro, n={len(gold_y)}):")
    print(f"  acurácia EP1: {correct_ep1.mean():.4f} | acurácia EP2: {correct_ep2.mean():.4f}")
    print(f"  EP1 certo / EP2 errado: {m['b']} | EP1 errado / EP2 certo: {m['c']}")
    if m["stat"] is not None:
        print(f"  estatística McNemar = {m['stat']:.4f} ({m['metodo']}), p = {m['p']:.4g}")
    else:
        print(f"  p = {m['p']:.4g} ({m['metodo']})")
    print(f"  diferença estatisticamente significativa a p<0,05: {'SIM' if m['p'] < 0.05 else 'NÃO'}")


if __name__ == "__main__":
    main(sys.argv[1:] or default_models())

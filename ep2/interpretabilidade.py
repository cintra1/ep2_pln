"""Item 9 do relatorio: interpretacao global e local (eli5) com um classificador simples, e
material para a analise manual de erros (usa as previsoes do EP2 no teste1, comparadas ao ouro).

Uso: python interpretabilidade.py
Gera: ep2/interpretabilidade_saida.txt (texto para o relatorio) e
      entrega/relatorio/eli5_pesos_globais.html, entrega/relatorio/eli5_exemplos_locais.html
"""
import os

import eli5
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import LABELS, ROOT, load_data

X, y, X_test, _ = load_data()
OUT_DIR = os.path.join(ROOT, "entrega", "relatorio")
os.makedirs(OUT_DIR, exist_ok=True)
TXT_OUT = os.path.join(os.path.dirname(__file__), "interpretabilidade_saida.txt")

# ---------------------------------------------------------------- classificador simples p/ eli5
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=5)),
    ("clf", LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")),
])
pipe.fit(X, y)
vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]

lines = []


def log(s=""):
    print(s)
    lines.append(s)


log("=" * 80)
log("CLASSIFICADOR SIMPLES PARA INTERPRETACAO: TF-IDF (1-2 gramas, min_df=5) + regressao")
log("logistica multinomial (C=0.3, class_weight=balanced), treinado nos 20.092 textos de treino.")
log(f"Acuracia desse classificador simples no proprio treino (nao e o modelo final): "
    f"{np.mean(pipe.predict(X) == y):.4f}")
log("=" * 80)

# ---------------------------------------------------------------- (i) interpretacao global
exp_global = eli5.explain_weights(clf, vec=vec, top=15, target_names=LABELS)
with open(os.path.join(OUT_DIR, "eli5_pesos_globais.html"), "w", encoding="utf-8") as f:
    f.write(eli5.format_as_html(exp_global))

log("\n(i) INTERPRETACAO GLOBAL - palavras/bigramas com maior peso por classe (eli5.explain_weights)\n")
df_global = eli5.format_as_dataframe(exp_global)
for target in LABELS:
    sub = df_global[df_global["target"] == target]
    top_pos = sub[sub["weight"] > 0].sort_values("weight", ascending=False).head(12)
    top_neg = sub[sub["weight"] < 0].sort_values("weight").head(12)
    log(f"--- classe {target} ---")
    log("  a favor: " + ", ".join(f"{r.feature} ({r.weight:+.2f})" for r in top_pos.itertuples()))
    log("  contra : " + ", ".join(f"{r.feature} ({r.weight:+.2f})" for r in top_neg.itertuples()))
    log("")

# ---------------------------------------------------------------- (ii) interpretacao local
pred_simple = pipe.predict(X_test)
proba_simple = pipe.predict_proba(X_test)
rng = np.random.default_rng(0)
# um exemplo por classe prevista, com confianca alta (bom para interpretar)
idx_local = []
for c in range(3):
    cand = np.where(pred_simple == c)[0]
    if len(cand):
        idx_local.append(cand[np.argmax(proba_simple[cand, c])])

log("\n(ii) INTERPRETACAO LOCAL - um exemplo do teste1 por classe prevista (eli5.explain_prediction)\n")
htmls = []
for i in idx_local:
    text = X_test.iloc[i]
    exp_local = eli5.explain_prediction(clf, text, vec=vec, target_names=LABELS)
    htmls.append(eli5.format_as_html(exp_local))
    log(f"--- teste1[{i}], previsto {LABELS[pred_simple[i]]} (p={proba_simple[i, pred_simple[i]]:.3f}) ---")
    log(f"  texto: {text[:300]!r}")
    log("")
with open(os.path.join(OUT_DIR, "eli5_exemplos_locais.html"), "w", encoding="utf-8") as f:
    f.write("<hr>".join(htmls))

# ---------------------------------------------------------------- (iii) material p/ analise manual de erros
log("\n(iii) MATERIAL PARA ANALISE MANUAL DE ERROS - previsoes do EP2 (stack final) x ouro no teste1\n")
test1_rot = pd.read_excel(os.path.join(ROOT, "ep2", "ep2_test1_rotulado.xlsx"))
gold = pd.read_excel(os.path.join(ROOT, "test1_ouro_mcnemar.xlsx"))
assert test1_rot["resp_text"].fillna("").astype(str).equals(gold["resp_text"].fillna("").astype(str))
errado = test1_rot["clarity"] != gold["clarity"]
log(f"Erros do EP2 no teste1 (ouro): {errado.sum()} de {len(gold)} ({errado.mean():.1%})")

erros = pd.DataFrame({
    "texto": test1_rot.loc[errado, "resp_text"],
    "previsto": test1_rot.loc[errado, "clarity"],
    "correto": gold.loc[errado, "clarity"],
})
amostra = erros.sample(min(12, len(erros)), random_state=0)
for r in amostra.itertuples():
    log(f"- previsto={r.previsto} correto={r.correto} | {str(r.texto)[:280]!r}")

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nSalvo: {TXT_OUT}")
print(f"HTML eli5: {OUT_DIR}")

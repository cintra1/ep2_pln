# EP2 - Classificação da clareza de respostas do e-SIC

ACH2118 Introdução ao Processamento de Língua Natural - EACH/USP

Stacking de 13 modelos de base (regressão logística multinomial sobre as probabilidades fora da dobra):
os 7 modelos clássicos de TF-IDF/Word2Vec e o BERTimbau-base + DAPT do EP1, mais um NB-SVM e quatro
modelos neurais novos (GovBERT-BR, Legal-BERTimbau-base, BERTikal, BERTimbau-large) e uma nova rodada do
BERTimbau-base + DAPT com outra semente. Resultado: 0,4782 de acurácia OOF (5 dobras, 20.092 textos),
contra 0,4708 do EP1 (McNemar no teste1, p = 0,047).

## Estrutura do repositório

- `train.xlsx`, `test1.xlsx`, `test2.xlsx`, `test1_ouro_mcnemar.xlsx`: os dados (os mesmos do EP1 + o
  gabarito do teste1 + o teste2, divulgados no enunciado do EP2).
- `ep1_test1_rotulado.xlsx`: a planilha de teste1 entregue no EP1 (usada só para comparar os dois modelos).
- `ep1_pln/`: cache dos modelos clássicos e das 5 dobras do BERTimbau+DAPT originais do EP1
  (`bert_fold0..4.npz`, `modelos_classicos.npz`). O checkpoint de pesos do DAPT em si não está aqui (419
  MB) porque o notebook do Colab treina esse encoder do zero automaticamente se não o encontrar.
- `ep2/`: todo o pipeline do EP2.
  - `common.py`, `stack_lib.py`, `dup.py`: utilitários compartilhados (carregar dados, dobras, suavização
    de Dirichlet para textos duplicados).
  - `importar_ep1.py`: converte o cache do EP1 (`ep1_pln/`) para o formato de previsões do EP2.
  - `novos.py`: modelos clássicos novos do EP2 (NB-SVM e outros experimentais) nas mesmas 5 dobras.
  - `backfill_test2_classicos.py`: gera as previsões de teste2 dos modelos herdados do EP1 (que nunca
    tinham visto o teste2, pois ele não existia no EP1).
  - `importar_gpu.py`: converte os resultados do notebook do Colab (`gpu_runs/*.npz`) para o formato do
    stacking.
  - `stack_final.py`: treina o meta-modelo, escolhe os hiperparâmetros por validação aninhada, gera a
    planilha de teste1 (comparação) e a planilha oficial de teste2, e roda o teste de McNemar contra o EP1.
  - `interpretabilidade.py`: análise eli5 (global e local) e a amostra de erros usada no relatório.
  - `ablation.py`, `greedy_backward.py`: estudo de robustez do ensemble (reproduz a comparação entre o
    stacking de 13 modelos e o subconjunto de 9 encontrado por eliminação reversa, discutida no relatório).
  - `preds/`: cache das previsões de cada modelo (OOF + teste1 + teste2), já calculado.
  - `gpu_runs/`: saída bruta do notebook do Colab para cada modelo neural, já baixada.
- `colab/ep2_colab.ipynb`: notebook único que treina qualquer um dos modelos neurais (fine-tuning
  completo ou embeddings congelados), escolhido por um `PRESET` na célula de configuração.

## Link do Colab

https://colab.research.google.com/drive/1xbXmNZq-LVzhvnkWQpdm3u2NJqWcRogc?usp=sharing

## Passo a passo para reproduzir do zero

O repositório já vem com o cache de todas as etapas (em `ep1_pln/`, `ep2/preds/` e `ep2/gpu_runs/`), então
o passo 5 sozinho já reproduz o resultado final. Os passos 1 a 4 mostram como cada parte desse cache foi
gerada, caso quaisquer dados ou modelos mudem.

1. Instalar as dependências: `pip install -r requirements.txt` (modelos clássicos e stacking, em CPU).
2. Modelos herdados do EP1: `cd ep2 && python importar_ep1.py` (lê `ep1_pln/`).
3. Modelos clássicos novos do EP2: `python novos.py nbsvm` e depois `python backfill_test2_classicos.py`
   (gera, para os modelos herdados do EP1 e o NB-SVM, as previsões do teste2, que não existiam no EP1).
4. Modelos neurais (GovBERT-BR, Legal-BERTimbau-base, BERTikal, BERTimbau-large, BERTimbau-base+DAPT com
   semente nova): abrir `colab/ep2_colab.ipynb` no Google Colab (link acima), selecionar
   *Ambiente de execução > Alterar o tipo de ambiente de execução > GPU T4*, escolher o `PRESET` na célula
   de configuração (`govbert_br`, `bertikal`, `legalbertimbau`, `bertimbau_large`,
   `bertimbau_base_dapt_s7`, nessa ordem) e *Executar tudo*. O notebook mantém os dados e os checkpoints
   no Google Drive (`MyDrive/ep2_pln` e `MyDrive/ep1_pln`): se não encontrar algo, treina do zero; se
   encontrar, reaproveita. Repetir para cada preset, baixar o `.npz` gerado e salvar em `ep2/gpu_runs/`.
5. Rodar `python importar_gpu.py` (converte os arquivos de `gpu_runs/` para o formato do stacking) e
   depois `python stack_final.py`. Esse script imprime a acurácia OOF de cada um dos 13 modelos e do
   stacking final, o teste de McNemar contra o EP1, e gera:
   - `ep2/ep2_test1_rotulado.xlsx` (comparação com o EP1, nunca é entrega)
   - `../entrega/teste2_resultado/ep2_test2_rotulado.xlsx` (a entrega oficial do EP2)
6. Opcional: `python interpretabilidade.py` reproduz a análise eli5 e a amostra de erros do relatório;
   `python ablation.py` e `python greedy_backward.py` reproduzem o estudo de robustez do ensemble (por que
   ficamos com os 13 modelos em vez de um subconjunto menor com OOF um pouco maior).

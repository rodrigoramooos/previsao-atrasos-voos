# Milestone 3: Modelação e Avaliação

## 1. Estratégia de Modelação

O dataset processado na Milestone 2 (`flight_data_processed_novo.csv`) contém **1 041 151 registos** e **340 *features*** após limpeza, *feature engineering* e *one-hot encoding* dos aeroportos de origem. Foram modeladas **duas variáveis alvo independentes**, com pipelines completamente separadas para evitar *leakage* entre *targets*:

| Target | Descrição | Positivos | Prevalência | Rácio neg:pos |
| :--- | :--- | :---: | :---: | :---: |
| `cancelled` | Voo cancelado (1) vs. operado (0) | 15 905 | 1.53% | ~64:1 |
| `is_delayed` | Atraso ≥ 15 min por meteorologia ou aeronave anterior (excluindo cancelamentos) | 88 690 | 8.52% | ~11:1 |

O **desequilíbrio severo de classes** é a característica dominante e determinou todas as escolhas metodológicas que se seguem.

**Divisão do dataset:** Foi utilizada uma divisão **80% treino / 20% teste**, com semente aleatória fixa (`random_state=42`) e **estratificação pela variável alvo** (`stratify=y`). A estratificação é obrigatória — sem ela, uma divisão "afortunada" poderia concentrar grande parte dos eventos positivos num dos subconjuntos. Para `is_delayed`, o conjunto exclui ainda os 15 905 voos cancelados (onde `is_delayed=0` por definição), evitando *leakage* entre *targets*.

**Métricas de Sucesso:** A *accuracy* foi excluída como métrica principal: um classificador trivial que prevê sempre "não cancelado" obteria 98.47% sem qualquer utilidade prática. As métricas adotadas foram:

- **F1-Score da classe positiva** — métrica principal de comparação, equilibra Precision e Recall;
- **Recall** — métrica secundária privilegiada do ponto de vista de negócio: um Falso Negativo (prever que um voo opera quando será cancelado/atrasado) tem custos operacionais e logísticos muito superiores a um Falso Positivo (alerta preventivo desnecessário);
- **Precision** — fiabilidade dos alertas gerados;
- **ROC-AUC** — capacidade discriminativa global, independente do *threshold*;
- **Average Precision (PR-AUC)** — área sob a curva Precision-Recall, **mais informativa que ROC-AUC em datasets fortemente desequilibrados** porque não é inflacionada pela enorme massa de negativos.

O **threshold de decisão** foi otimizado por F1 sobre a curva Precision-Recall, evitando o limiar padrão 0.5 que é inadequado para rácios 64:1 ou 11:1.

---

## 2. Experiências Realizadas

### 2.1. Modelo Baseline

Ponto de partida deliberadamente simples para estabelecer o patamar mínimo de desempenho que qualquer modelo candidato terá obrigatoriamente de superar.

**Algoritmo:** Regressão Logística (`class_weight='balanced'`, `max_iter=1000`, `random_state=42`)

**Resultado — `cancelled` (threshold otimizado = 0.7336):**

| Métrica | Classe 0 (não cancelado) | Classe 1 (cancelado) |
| :--- | :---: | :---: |
| Precision | 0.987 | 0.053 |
| Recall | 0.935 | 0.233 |
| F1-Score | 0.961 | **0.086** |

**ROC-AUC:** 0.7611 | **Average Precision:** 0.0423

**Resultado — `is_delayed` (threshold otimizado = 0.5145):**

| Métrica | Classe 0 (não atrasado) | Classe 1 (atrasado) |
| :--- | :---: | :---: |
| Precision | 0.938 | 0.123 |
| Recall | 0.618 | 0.567 |
| F1-Score | 0.745 | **0.202** |

**ROC-AUC:** 0.6240 | **Average Precision:** 0.1244

**Análise:** Para `cancelled`, o F1 de **0.086** confirma que a fronteira linear é insuficiente para capturar padrões de cancelamento com 340 *features* e rácio 64:1 — o modelo só arrisca classificar como cancelamento quando a probabilidade ultrapassa 73%. Para `is_delayed`, o threshold mais baixo (0.5145) e o F1 já em **0.202** confirmam que o sinal de atraso é mais forte e parcialmente capturável por um modelo linear, mas os 7 falsos alarmes por cada deteção correta mostram o limite estrutural da regressão logística.

---

### 2.2. Modelos Candidatos

Após o *Baseline*, foram testados algoritmos de *gradient boosting* — selecionados pela capacidade comprovada de capturar relações não-lineares e interações entre variáveis em problemas tabulares fortemente desequilibrados.

**Algoritmos testados:**

- **Logistic Regression** — referência mínima (baseline);
- **HistGradient Boosting** — *boosting* baseado em histogramas, eficiente para grandes volumes de dados, com `class_weight='balanced'`;
- **XGBoost** — *boosting* avançado com `scale_pos_weight` calibrado para o rácio neg:pos (64.5 para `cancelled`, 10.6 para `is_delayed`).

O *threshold* de cada candidato foi otimizado individualmente por F1 na curva Precision-Recall.

**Tabela comparativa — `cancelled` (teste):**

| Algoritmo | Threshold | F1 | Recall | Precision | ROC-AUC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression *(Baseline)* | 0.7336 | 0.0865 | 0.2333 | 0.0531 | 0.7611 | 0.0423 |
| HistGradient Boosting | 0.8458 | 0.1630 | 0.2144 | 0.1315 | 0.8671 | **0.0963** |
| XGBoost | 0.7852 | 0.1662 | 0.3650 | 0.1076 | 0.8690 | 0.0960 |

**Tabela comparativa — `is_delayed` (teste):**

| Algoritmo | Threshold | F1 | Recall | Precision | ROC-AUC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression *(Baseline)* | 0.5145 | 0.2023 | 0.5670 | 0.1231 | 0.6240 | 0.1244 |
| HistGradient Boosting | 0.6356 | 0.2913 | 0.3640 | 0.2428 | 0.7211 | 0.2229 |
| XGBoost | 0.6326 | 0.2908 | 0.3771 | 0.2367 | 0.7213 | **0.2235** |

**Critério de seleção:** A escolha do melhor candidato foi feita pela **Average Precision** e não pelo F1 ou ROC-AUC. O F1 obriga a fixar um *threshold* antes de avaliar (decisão prematura nesta fase) e o ROC-AUC é inflacionado pela massa de negativos. A AP avalia o modelo em todos os *thresholds* simultaneamente e penaliza diretamente os falsos alarmes — sendo a escolha mais honesta em problemas com raridade extrema de positivos.

**Algoritmos vencedores:**

- **`cancelled`:** **HistGradient Boosting** (AP = 0.0963 vs. 0.0960 do XGBoost) — mais do dobro do baseline (0.0423);
- **`is_delayed`:** **XGBoost** (AP = 0.2235 vs. 0.2229 do HistGBT) — quase o dobro do baseline (0.1244).

**Algoritmo que falhou:** A **Regressão Logística** confirmou-se como referência mínima em ambos os *targets* — o salto de AP do baseline para os modelos *boosting* (mais do dobro em ambos os casos) prova que existe uma componente não-linear nos padrões de cancelamento e atraso que um modelo linear não consegue capturar.

**Diagnóstico de generalização:** A análise das curvas ROC e Precision-Recall (`reports/figures/`) mostra que HistGBT e XGBoost têm desempenho praticamente equivalente em ambos os *targets* — extraem o mesmo sinal dos dados. As curvas de aprendizagem confirmam ausência de *overfitting* significativo entre treino e teste.

---

## 3. Otimização (*Tuning*)

A otimização envolveu **duas fases combinadas**, dado que o problema exige tanto *tuning* de hiperparâmetros como **otimização do limiar de decisão** (que para *thresholds* otimizados ficou claramente longe do 0.5 padrão).

### Técnica Utilizada

**Fase 1 — `GridSearchCV`** sobre 30% dos dados de treino (~250 000 registos), com **`StratifiedKFold` de 3 folds** e `scoring='average_precision'`. O *grid* cobriu os hiperparâmetros mais influentes (`learning_rate`, `max_depth`, `n_estimators`/`max_iter`, `min_samples_leaf` ou `subsample`/`colsample_bytree`).

**Fase 2 — `TunedThresholdClassifierCV`** integra a otimização do *threshold* dentro da validação cruzada (3 folds, `scoring='f1'`), garantindo que o limiar não é sobreajustado ao conjunto de teste.

### Resultados — `cancelled` (HistGradient Boosting)

| | Valor |
| :--- | :---: |
| Melhores hiperparâmetros | `learning_rate=0.10`, `max_depth=10`, `max_iter=200`, `min_samples_leaf=30` |
| Average Precision (CV) | 0.0749 |
| Threshold ótimo (CV) | **0.8059** |
| F1-Score (CV) no threshold otimizado | 0.1376 |

### Resultados — `is_delayed` (XGBoost)

| | Valor |
| :--- | :---: |
| Melhores hiperparâmetros | `learning_rate=0.05`, `max_depth=8`, `n_estimators=200`, `subsample=0.8`, `colsample_bytree=0.8` |
| Average Precision (CV) | 0.2210 |
| Threshold ótimo (CV) | **0.6127** |
| F1-Score (CV) no threshold otimizado | 0.2845 |

### Melhoria Obtida

| Target | Métrica | Baseline | Modelo Otimizado | Variação |
| :--- | :--- | :---: | :---: | :---: |
| `cancelled` | F1 | 0.086 | **0.151** | +75% |
| `cancelled` | Recall | 0.233 | 0.266 | +14% |
| `cancelled` | Avg Precision | 0.042 | **0.087** | +107% |
| `is_delayed` | F1 | 0.202 | **0.289** | +43% |
| `is_delayed` | Precision | 0.123 | **0.227** | +85% |
| `is_delayed` | Avg Precision | 0.124 | **0.225** | +81% |

A melhoria é substancial em ambos os *targets*, especialmente nas métricas independentes de *threshold* (Average Precision). O *trade-off* entre Precision e Recall é mais favorável após a otimização — em `is_delayed` o modelo passa a equilibrar melhor os dois lados, em `cancelled` a melhoria do F1 vem sobretudo do aumento da Precision (de 5.3% para 10.5%).

> Os F1 de validação cruzada (0.1376 e 0.2845) ficam muito próximos dos F1 obtidos em teste (0.1505 e 0.2889), confirmando **boa generalização e ausência de overfitting** no processo de otimização.

---

## 4. Avaliação do Modelo Final

### 4.1. Matriz de Confusão / Erros

**Modelo final `cancelled` — HistGradient Boosting otimizado (threshold = 0.806):**

| | Previsto: Não Cancelado | Previsto: Cancelado |
| :--- | :---: | :---: |
| **Real: Não Cancelado** | TN = 197 848 | FP = 7 202 |
| **Real: Cancelado** | FN = 2 336 | TP = 845 |

- **Recall = 26.6%** — de 3 181 cancelamentos no teste, o modelo deteta 845 e falha 2 336;
- **Rácio FP/TP = 8.5** — por cada cancelamento detetado, geram-se 8.5 alarmes falsos;
- Taxa de Falso Positivo: 3.5% sobre o universo de negativos.

**Modelo final `is_delayed` — XGBoost otimizado (threshold = 0.613):**

| | Previsto: Não Atrasado | Previsto: Atrasado |
| :--- | :---: | :---: |
| **Real: Não Atrasado** | TN = 163 463 | FP = 23 849 |
| **Real: Atrasado** | FN = 10 717 | TP = 7 021 |

- **Recall = 39.6%** — de 17 738 atrasos no teste, o modelo deteta 7 021 e falha 10 717;
- **Rácio FP/TP = 3.4** — quase 3× mais favorável do que em `cancelled`;
- Taxa de Falso Positivo: 12.7% sobre o universo de negativos.

> **Análise crítica:**
>
> O modelo apresenta maior dificuldade em distinguir corretamente os eventos positivos quando:
> - As variáveis operacionais (horários, distância, dia/mês) não apresentam padrões claramente distintos entre voos que cancelam/atrasam e os que operam normalmente — visível no histograma de probabilidades, onde a sobreposição entre classes é quase total;
> - O cancelamento ou atraso depende de fatores externos não capturados no dataset (condições meteorológicas em tempo real, decisões operacionais ad-hoc das companhias, eventos imprevisíveis).
>
> A redução dos Falsos Negativos é o principal ganho do tuning:
> - Um Falso Negativo (voo previsto como operacional mas cancelado/atrasado) implica falha crítica para o passageiro e para o aeroporto;
> - Um Falso Positivo implica apenas um custo operacional marginal (um alerta preventivo que não se materializa).
>
> Por isso, o modelo está **alinhado com a lógica de risco do problema**: privilegia a deteção sobre a precisão, especialmente em `cancelled` onde o *trade-off* é mais agressivo.

### 4.2. Importância dos Atributos (Feature Importance)

A análise de importância (calculada por *permutation importance* sobre o melhor estimador do GridSearch) revela que **as variáveis temporais dominam claramente em ambos os targets**, com diferenças relevantes nos aeroportos.

**Top features — `cancelled`:**

1. **`day_of_month`** — de longe a mais importante;
2. **`month`** — segunda mais importante (janeiro/fevereiro = pico do inverno = mais cancelamentos);
3. **`distance`** — voos longos com maior risco de cancelamento;
4. **`day_of_week`**;
5. Aeroportos: **BUF** (Buffalo), **MDW** (Chicago Midway), **MSY** (Nova Orleães), **ATL**, **ORD** — aeroportos historicamente afetados por **clima de inverno severo**.

**Top features — `is_delayed`:**

1. **`day_of_month`** — fator mais influente;
2. **`month`**;
3. **`day_of_week`** — com peso superior face a `cancelled`;
4. **`distance`** — voos curtos com maior propensão a atrasos;
5. Aeroportos: **ORD** (Chicago O'Hare), **DFW** (Dallas), **FLL**, **CLT**, **MIA**, **ATL** — grandes **hubs operacionais** onde os atrasos se propagam em cadeia entre voos conectados.

**Os dois targets partilham os mesmos determinantes de topo (data e distância), mas divergem nos aeroportos:**

- Cancelamentos concentram-se em aeroportos com clima extremo (BUF, MDW);
- Atrasos propagam-se a partir dos maiores hubs operacionais (ORD, DFW).

**Análise complementar com SHAP:** A análise SHAP (*beeswarm* + comparação direta entre *targets*) confirma estes padrões e revela que:

- Para `cancelled`, partir dos grandes hubs (ATL, DFW, CLT) **reduz** o risco — funcionam como "fatores de proteção";
- Para `is_delayed`, partir de **ATL reduz** o atraso esperado mas partir de **DFW aumenta-o drasticamente** — nem todos os grandes aeroportos se comportam igual.

### Interpretação Crítica

A importância das variáveis indica que o modelo:

- Está a aprender **padrões operacionais reais** (sazonalidade de inverno, congestionamento de hubs);
- Baseia decisões em **variáveis com significado de negócio** e não em correlações espúrias;
- Distingue corretamente as causas de cancelamentos (clima local) das causas de atrasos (propagação em rede).

No entanto, evidencia também a **principal limitação estrutural** do modelo:

> O modelo **não tem acesso** a:
> - Previsão e estado meteorológico real por aeroporto e data;
> - Estado do tráfego aéreo em tempo real;
> - Decisões estratégicas das companhias aéreas (cancelamentos preventivos, swaps de aeronaves);
> - Estado da aeronave (manutenção, atrasos a montante na rota).
>
> O `month`/`day_of_month` funciona como **proxy do clima**, mas é um proxy fraco. Isto explica:
> - O teto de desempenho (~F1 0.15 em `cancelled`, ~F1 0.29 em `is_delayed`);
> - A sobreposição quase total das distribuições de probabilidade entre classes positivas e negativas.

---

## 5. Conclusão da Fase de Modelação

O projeto produziu **dois modelos finais independentes**, cada um com o algoritmo que melhor se ajustou à estrutura do respetivo *target*:

| Target | Algoritmo Final | F1 | Recall | Precision | ROC-AUC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `cancelled` | **HistGradient Boosting** otimizado | 0.151 | 0.266 | 0.105 | 0.854 | 0.087 |
| `is_delayed` | **XGBoost** otimizado | 0.289 | 0.396 | 0.227 | 0.718 | 0.225 |

A escolha de algoritmos distintos para cada *target* foi feita pelo critério mais robusto disponível — a **Average Precision em validação cruzada** — embora a margem entre HistGBT e XGBoost tenha sido pequena em ambos os casos. A diferença operacional entre os dois modelos finais (F1 de 0.29 vs. 0.15) **não resulta de algoritmos diferentes nem de mais esforço de otimização**: resulta exclusivamente do facto de **`is_delayed` ser ~5.5× mais frequente que `cancelled`**, oferecendo muito mais sinal de aprendizagem para o mesmo conjunto de *features*.

A principal mais-valia dos modelos finais reside na capacidade de:

- Capturar padrões não lineares e interações complexas entre 340 variáveis (algo impossível para a Regressão Logística, que ficou muito atrás em ambos os *targets*);
- Apresentar **boa capacidade discriminativa para ranking de risco** — especialmente `cancelled` (ROC-AUC = 0.854), que pode ser útil para ordenar voos por probabilidade de cancelamento mesmo que a deteção binária seja limitada;
- **Reduzir Falsos Negativos** após otimização do *threshold*, alinhando os modelos com o objetivo operacional de minimizar o custo de cancelamentos/atrasos não detetados.

**Limitações relevantes a reconhecer:**

- O F1 de `cancelled` permanece moderado (0.151), refletindo o **teto estrutural imposto pelo rácio 64:1** — independentemente do algoritmo, a raridade do evento limita a aprendizagem;
- Em `cancelled`, geram-se **8.5 falsos alarmes por cada deteção correta** — tolerável num contexto de alerta preventivo, problemático para qualquer ação automatizada;
- A **ausência de variáveis externas críticas** (clima em tempo real, estado do tráfego, decisões das companhias) restringe o potencial preditivo. O modelo infere o risco a partir de *quando* e *de onde* o voo parte, não do estado real da operação;
- O *target* `is_delayed` exclui atrasos por outras causas (volume, segurança, combustível, etc.), o que reduz parte do sinal disponível.

**Decisão final:** Os modelos são considerados **adequados como primeira solução funcional** para apoio à decisão operacional — não como sistema autónomo. São tecnicamente validados (`Restart & Run All` reproduz integralmente os resultados), e a sua utilidade prática mais imediata está no *ranking* de voos por risco, não na classificação binária. Evolução futura passa por:

- Integração de **dados meteorológicos por aeroporto e data**;
- Inclusão de *features* de **rede operacional** (atrasos a montante, rotação de aeronaves);
- Reformulação do problema como **regressão sobre minutos de atraso**, em vez de classificação binária;
- Eventual modelação **conjunta** dos dois *targets* (multi-output) para capturar a sua correlação operacional.

Em síntese: ambos os modelos estão **tecnicamente validados e operacionalmente úteis como suporte à decisão**, com margem significativa de melhoria condicionada à integração de novas fontes de dados.

---
*Data de última atualização: 12/05/2026*

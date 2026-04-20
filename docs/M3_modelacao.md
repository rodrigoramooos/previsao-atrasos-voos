# Milestone 3: Modelação e Avaliação
## 1. Estratégia de Modelação
 
*O dataset processado contém 1 041 151 registos, com a variável alvo `cancelled` (binária: 0 — voo não cancelado; 1 — voo cancelado). Antes de avançar para os algoritmos, foram tomadas as seguintes decisões metodológicas:*
 
**Divisão do dataset:** Utilizámos uma divisão de **80% para treino** e **20% para teste**, com semente aleatória fixa (`random_state=42`) e **estratificação pela variável alvo** (`stratify=y`). A estratificação foi obrigatória dado o desequilíbrio severo do dataset — apenas **1.53% dos voos foram cancelados**, garantindo que ambos os subconjuntos preservam esta proporção e que os dados de avaliação estão totalmente isolados, prevenindo qualquer fuga de informação (*data leakage*).
 
**Métrica de Sucesso:** A métrica principal escolhida foi o **F1-Score para a classe `cancelled`**, pois o dataset é fortemente desequilibrado (98.47% vs. 1.53%) e a *accuracy* global seria enganosamente elevada mesmo num classificador trivial, ou seja, um modelo que previsse sempre "não cancelado" atingiria 98% de *accuracy* sem qualquer valor preditivo. Do ponto de vista de negócio, privilegiámos ainda o **Recall** como métrica secundária, dado que um **Falso Negativo**, isto significa, prever que um voo opera quando será cancelado. Este problema provoca custos operacionais e logísticos muito superiores a um Falso Positivo (um alerta desnecessário). O **AUC-ROC** foi utilizado como métrica de suporte para comparar a capacidade discriminativa global dos modelos ao longo de diferentes limiares de decisão.
 
---
 
## 2. Experiências Realizadas
 
### 2.1. Modelo Baseline
 
O ponto de partida simples, para definir o patamar mínimo de desempenho que qualquer modelo candidato terá obrigatoriamente de superar.
 
**Algoritmo:** Regressão Logística (`class_weight='balanced'`, `max_iter=1000`, `random_state=42`)
 
**Resultado:**
 
| Métrica | Classe 0 (não cancelado) | Classe 1 (cancelado) |
| :--- | :---: | :---: |
| Precision | 0.99 | 0.03 |
| Recall | 0.61 | 0.80 |
| F1-Score | 0.76 | 0.06 |


AUC-ROC - 0.7601 
 
**Análise:** O F1-Score de **0.06** para a classe de cancelamentos revela que o modelo Baseline é incapaz de gerar previsões positivas fiáveis. Este resultado era esperado: a Regressão Logística assume **relações lineares** entre as variáveis preditoras e a probabilidade de cancelamento. A *accuracy* global de 0.61, inferior ao classificador trivial (que ao prever sempre 0 atingiria 0.98), resulta do balanceamento de pesos que desloca as previsões em favor da classe minoritária, gerando muitos Falsos Positivos e degradando a Precisão para 0.03. Este modelo **falha o objetivo de negócio** e serve apenas como referencial de comparação mínimo para as fases seguintes.
 
---
 
### 2.2. Modelos Candidatos
 
*Após o Baseline, avançámos para algoritmos de maior complexidade, selecionados pela capacidade de capturar relações não-lineares e interações entre variáveis — limitação estrutural que eliminou a Regressão Logística.*
 
**Algoritmos selecionados e justificação:**
 
- **Árvore de Decisão** — modelo interpretável e simples de analisar; serve de referência intermédia entre o Baseline e os algoritmos ensemble.
- **Random Forest** — ensemble de árvores com amostragem aleatória de variáveis (*feature bagging*), reduzindo a variância face a uma árvore individual.
- **Extra Trees** (*Extremely Randomized Trees*) — variante do Random Forest com maior aleatoriedade nos pontos de divisão, frequentemente mais rápida e com boa capacidade de generalização.
Todos os modelos foram configurados com `class_weight='balanced_subsample'` (Random Forest e Extra Trees) ou `class_weight='balanced'` (Árvore de Decisão) para mitigar o desequilíbrio de classes, e com `random_state=42` para garantir reprodutibilidade.
 
**Tabela comparativa de desempenho (conjunto de teste):**
 
| Algoritmo | Parâmetros Base | Accuracy | Recall | F1-Score | AUC-ROC | Notas |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Regressão Logística *(Baseline)* | class_weight=balanced | 0.61 | 0.80 | 0.06 | 0.7601 | Referencial mínimo |
| Árvore de Decisão | max_depth=12, min_samples_leaf=20 | 0.71 | 0.8434 | 0.08 | 0.8369 | Melhor F1; gap treino-teste mínimo |
| Random Forest | n_estimators=120, max_depth=15 | 0.67 | 0.8419 | 0.07 | 0.8373 | Maior accuracy global; a aguardar métricas completas |
| Extra Trees | n_estimators=120, max_depth=15 | 0.67 | 0.8444 | 0.07 | 0.8341 | Desempenho semelhante ao Random Forest |
 
**O algoritmo que se destacou:** A **Árvore de Decisão** obteve o melhor F1-Score (0.08) e Recall (0.84) nesta fase inicial de comparação, com um gap treino-teste mínimo (F1: 0.0847 treino vs. 0.0815 teste), o que demonstra boa capacidade de generalização sem overfitting severo. O Random Forest e o Extra Trees apresentam *accuracy* superior, mas carecem ainda de análise aprofundada das métricas por classe.
 
**O algoritmo que falhou:** A **Regressão Logística** (Baseline) falhou inequivocamente, confirmando que a hipótese de linearidade é incompatível com a complexidade deste problema.
 
**Diagnóstico de generalização — treino vs. teste (Árvore de Decisão):**
 
| Conjunto | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Treino | 0.7093 | 0.0445 | 0.8804 | 0.0847 | 0.8648 |
| Teste | 0.7097 | 0.0428 | 0.8434 | 0.0815 | 0.8369 |
| Diferença | +0.0004 | +0.0017 | +0.0370 | +0.0032 | +0.0279 |
 
A diferença de F1-Score entre treino e teste é de apenas **0.003**, demonstrando a ausência de overfitting severo. O AUC-ROC em teste (0.8369) é substancialmente superior ao Baseline (0.7601), confirmando uma melhoria real na capacidade discriminativa do modelo.
 
**Curvas de aprendizagem (5-fold CV, F1-Score):**
 
| Conjunto | F1-Score médio final |
| :--- | :---: |
| Treino | 0.0859 |
| Validação | 0.0813 |
| Gap | 0.0046 |
 
As curvas de aprendizagem revelam um comportamento globalmente estável, com bandas de variância estreitas e convergência progressiva das curvas à medida que aumenta o volume de dados de treino. O gap reduzido entre treino e validação confirma a robustez do modelo e indica que o aumento de dados de treino contribui positivamente para a generalização. O F1-Score baixo em ambos os conjuntos (≈0.08) é um sinal de **subajuste moderado** da Árvore de Decisão com os parâmetros atuais.
 
---
## 3. Otimização (Tuning)
*Descrevam como melhoraram o melhor modelo.*
* **Técnica Utilizada:** (p/ex.: "Utilizámos GridSearchCV para ajustar os hiperparâmetros
`max_depth` e `learning_rate`.")
* **Melhoria obtida:** (p/ex.: "O F1-Score subiu de 0.85 para 0.88 após o ajuste.")
## 4. Avaliação do Modelo Final
### 4.1. Matriz de Confusão / Erros
*Analisem onde o modelo mais falha.*
> **Análise:** (p/ex.: "O modelo ainda confunde a Classe A com a Classe B em 10% dos casos devido
à semelhança nos atributos X e Y.")
### 4.2. Importância dos Atributos (Feature Importance)
*Quais as variáveis que o modelo considerou mais importantes para decidir?*
1. [Variável X]
2. [Variável Y]
## 5. Conclusão da Fase de Modelação
*Justifiquem por que razão este modelo está pronto (ou não) para ser apresentado como solução
final.*
---
*Data de última atualização: [20/04/2026]*

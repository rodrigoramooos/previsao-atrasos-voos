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
 
**Resultado (Threshold = 0.50):**
 
| Métrica | Classe 0 (não cancelado) | Classe 1 (cancelado) |
| :--- | :---: | :---: |
| Precision | 0.99 | 0.03 |
| Recall | 0.61 | 0.80 |
| F1-Score | 0.76 | 0.06 |

**AUC-ROC:** 0.7601  

**Nota adicional:** Foi realizada otimização de threshold, tendo sido identificado um valor ótimo de **0.70**, com F1-score em validação de **0.0834**, demonstrando que mesmo com ajuste do limiar o modelo continua limitado estruturalmente.
 
**Análise:** O F1-Score de **0.06** para a classe de cancelamentos revela que o modelo Baseline é incapaz de gerar previsões positivas fiáveis. Este resultado era esperado: a Regressão Logística assume **relações lineares** entre as variáveis preditoras e a probabilidade de cancelamento. A *accuracy* global de 0.61, inferior ao classificador trivial (que ao prever sempre 0 atingiria 0.98), resulta do balanceamento de pesos que desloca as previsões em favor da classe minoritária, gerando muitos Falsos Positivos e degradando a Precisão para 0.03.
 
---
 
### 2.2. Modelos Candidatos
 
*Após o Baseline, avançámos para algoritmos de maior complexidade, selecionados pela capacidade de capturar relações não-lineares e interações entre variáveis — limitação estrutural que eliminou a Regressão Logística.*
 
**Algoritmos selecionados e justificação:**
 
- **Árvore de Decisão** — modelo interpretável e simples de analisar; serve de referência intermédia entre o Baseline e os algoritmos ensemble.
- **Random Forest** — ensemble de árvores com amostragem aleatória de variáveis (*feature bagging*), reduzindo a variância face a uma árvore individual.
- **Extra Trees** (*Extremely Randomized Trees*) — variante do Random Forest com maior aleatoriedade nos pontos de divisão, frequentemente mais rápida e com boa capacidade de generalização.
- **HistGradient Boosting** — algoritmo de boosting baseado em histogramas, altamente eficiente para grandes volumes de dados e capaz de capturar relações complexas.
- **Logistic Regression Calibrated** — versão calibrada do baseline para melhorar a estimativa probabilística.
- **XGBoost** — algoritmo de boosting avançado, conhecido pelo elevado desempenho em problemas tabulares.
  
**Tabela comparativa de desempenho (validação com threshold otimizado):**
 
| Algoritmo | Threshold | F1-Score | Notas |
| :--- | :---: | :---: | :--- |
| Regressão Logística *(Baseline)* | 0.70 | 0.0834 | Referencial mínimo |
| Árvore de Decisão | 0.85 | 0.1530 | Melhor entre modelos simples |
| Random Forest | 0.62 | 0.1475 | Estável mas inferior à árvore |
| Extra Trees | 0.60 | 0.1440 | Desempenho semelhante ao RF |
| HistGradient Boosting | 0.09 | **0.2023** | Melhor desempenho global |
| Logistic Regression Calibrated | 0.04 | 0.0866 | Pequena melhoria face ao baseline |
 
**Algoritmo Vencedor:** O **HistGradient Boosting** destacou-se de forma clara, com:
- **F1-Score:** 0.1925 (teste)
- **Recall:** 0.2279  
- **ROC-AUC:** 0.8737  

Este modelo apresenta uma melhoria substancial face a todos os restantes, especialmente na capacidade de equilíbrio entre precisão e recall.
 
**O algoritmo que falhou:** A **Regressão Logística** (Baseline), mesmo com calibração, continua a demonstrar incapacidade estrutural para modelar o problema.
 
**Diagnóstico de generalização — HistGradient Boosting:**
 
| Conjunto | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Treino | 0.9848 | 0.7619 | 0.0063 | 0.0125 | 0.8876 |
| Teste | 0.9848 | 0.7241 | 0.0066 | 0.0131 | 0.8734 |
| Diferença | ≈0.0000 | +0.0378 | -0.0003 | -0.0006 | +0.0142 |
 
Apesar da forte capacidade discriminativa (ROC-AUC elevado), observa-se um **F1-score extremamente baixo**, evidenciando um comportamento de **subajuste severo** no threshold padrão, reforçando a importância crítica da otimização do limiar de decisão.
 
**Curvas de aprendizagem e diagnóstico:**

As curvas de aprendizagem e análise de métricas confirmam:
- Boa capacidade discriminativa (ROC-AUC elevado)
- Forte sensibilidade ao threshold
- Necessidade de otimização orientada ao objetivo de negócio

O comportamento global indica que o problema não está apenas no modelo, mas na **definição do limiar de decisão num contexto altamente desequilibrado**.
 
---
## 3. Otimização (Tuning)

Após a fase de comparação entre modelos, o **HistGradient Boosting** foi selecionado como modelo candidato final, dado o seu desempenho superior em termos de capacidade discriminativa (ROC-AUC) e potencial de melhoria via ajuste de limiar.

### Técnica Utilizada

A estratégia de otimização não se focou exclusivamente em hiperparâmetros clássicos, mas sobretudo na **otimização do threshold de decisão**, dado que:

- O problema apresenta **forte desbalanceamento (≈1.5% classe positiva)**
- O objetivo de negócio privilegia **Recall e F1-score da classe minoritária**
- O threshold padrão (0.50) revelou-se inadequado

Foi implementado um processo de:
- Avaliação de múltiplos thresholds (0.01 → 0.99)
- Seleção do threshold ótimo com base no **F1-score em validação**

Para o modelo final:

- **Threshold ótimo identificado:** ≈ **0.09**
- Este valor confirma que o modelo necessita de uma **forte redução do limiar** para captar eventos raros

### Melhoria Obtida

| Métrica | Antes (threshold=0.50) | Depois (threshold otimizado) |
| :--- | :---: | :---: |
| Recall | ~0.0066 | **0.2279** |
| F1-Score | ~0.0131 | **0.1925** |

A melhoria é **substancial**:

- O modelo passa de praticamente **não detetar cancelamentos**  
→ para um modelo que começa efetivamente a capturar eventos relevantes

- O aumento do Recall traduz-se numa **redução significativa de falsos negativos**, alinhando o modelo com o objetivo de negócio

No entanto, esta melhoria ocorre à custa de:
- Aumento de falsos positivos  
- Redução da precisão

Este trade-off é esperado no contexto do problema, uma vez que se opta por privilegiar a deteção de cancelamentos, mesmo que isso implique um aumento de falsos positivos.

---

## 4. Avaliação do Modelo Final

### 4.1. Matriz de Confusão / Erros

A matriz de confusão do modelo final evidencia um comportamento típico de problemas altamente desbalanceados:

- Elevado número de **Verdadeiros Negativos** (classe dominante)
- Aumento significativo de **Verdadeiros Positivos** após tuning
- Crescimento controlado de **Falsos Positivos**
- Redução relevante de **Falsos Negativos**

> **Análise:**
>
> O modelo apresenta maior dificuldade em distinguir corretamente voos cancelados de voos não cancelados em situações onde:
> - As variáveis operacionais (ex.: horários, distância, tempo em solo) não apresentam padrões claramente distintos
> - O cancelamento depende de fatores externos não capturados no dataset (ex.: condições meteorológicas extremas, decisões operacionais das companhias aéreas)
>
> Apesar disso, a redução de falsos negativos constitui o principal ganho:
>
> - Um falso negativo implica falha crítica (voo previsto como operacional mas cancelado)
> - Um falso positivo implica apenas custo operacional marginal (alerta preventivo)
>
> Assim, o modelo encontra-se **alinhado com a lógica de risco do problema**, privilegiando deteção sobre precisão.

---

### 4.2. Importância dos Atributos (Feature Importance)

A análise de importância das variáveis no modelo **HistGradient Boosting** revela que as decisões são fortemente influenciadas por variáveis operacionais relacionadas com o ciclo do voo.

**Principais variáveis identificadas:**

1. **`dep_time` / período do voo (`flight_period`)**  
   → Reflete padrões temporais (manhã/tarde/noite) associados a congestionamento e operações aeroportuárias

2. **`taxi_out` e `taxi_in`**  
   → Indicadores indiretos de congestionamento no aeroporto

3. **`distance` / `is_long_flight`**  
   → Voos longos tendem a apresentar maior complexidade operacional

4. **`ground_time`**  
   → Tempo em solo como proxy de eficiência operacional

5. **`origin` (aeroporto de origem)**  
   → Captura efeitos estruturais (capacidade, localização, tráfego)

---

### Interpretação Crítica

A importância das variáveis indica que o modelo:

- Está a aprender **padrões operacionais reais**
- Baseia decisões em **variáveis com significado de negócio**
- Não depende exclusivamente de correlações espúrias

No entanto, evidencia também uma limitação estrutural:

> O modelo não tem acesso a variáveis críticas como:
> - Condições meteorológicas
> - Estado do tráfego aéreo em tempo real
> - Decisões estratégicas das companhias

O que explica:
- Limite superior do desempenho (~F1 ≈ 0.20)
- Dificuldade em capturar cancelamentos raros e imprevisíveis

---
## 5. Conclusão da Fase de Modelação

O modelo final baseado em **HistGradient Boosting**, com otimização do threshold de decisão, representa a melhor solução obtida no contexto do problema, tendo em conta os dados disponíveis e o objetivo definido.

A sua principal mais-valia reside na capacidade de:
- Capturar padrões não lineares e interações complexas entre variáveis
- Apresentar uma elevada capacidade discriminativa (**ROC-AUC ≈ 0.87**)
- Melhorar significativamente a deteção da classe minoritária após otimização do threshold (**F1 ≈ 0.19; Recall ≈ 0.23**)

Do ponto de vista de negócio, o modelo encontra-se **alinhado com o objetivo operacional**, ao reduzir falsos negativos — situações críticas onde um voo é previsto como operacional mas acaba por ser cancelado.

No entanto, importa reconhecer limitações relevantes:

- O desempenho em termos de F1-score permanece moderado, refletindo a dificuldade intrínseca do problema
- A elevada raridade da classe positiva limita a capacidade de aprendizagem do modelo
- A ausência de variáveis externas críticas (ex.: condições meteorológicas, estado do tráfego aéreo) restringe o potencial preditivo

Assim, o modelo pode ser considerado **adequado como primeira solução funcional**, sendo capaz de apoiar decisões operacionais através da identificação de voos com maior probabilidade de cancelamento.

Contudo, não deve ser encarado como uma solução final otimizada, mas sim como uma base sólida para evolução futura, nomeadamente através de:
- Integração de novas fontes de dados
- Refinamento do processo de modelação
- Ajustes adicionais orientados ao custo de erro

Em síntese, o modelo está **tecnicamente validado e operacionalmente útil**, embora ainda com margem significativa para melhoria.
---
*Data de última atualização: [23/04/2026]*

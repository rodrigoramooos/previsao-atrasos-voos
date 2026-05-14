# Milestone 1 — Iniciação e Definição do Projeto

## 1. Descrição do Problema

O transporte aéreo comercial é um setor de elevada complexidade operacional, onde cancelamentos e atrasos de voos geram impactos financeiros significativos para companhias aéreas, aeroportos e passageiros. Nos Estados Unidos, dados do *Bureau of Transportation Statistics* indicam que perturbações operacionais custam anualmente milhares de milhões de dólares à indústria, não só em compensações diretas, mas também em custos de realojamento, afetação de tripulações e deterioração da experiência do passageiro.

O problema identificado consiste na ausência de instrumentos preditivos capazes de antecipar, antes da partida, se um determinado voo irá ser cancelado ou sofrer um atraso significativo. A gestão reativa destas perturbações — ou seja, agir apenas após a ocorrência — é operacionalmente ineficaz e economicamente prejudicial. Um sistema de alerta precoce baseado em dados históricos permitiria às companhias aéreas e aeroportos agir com antecedência, reduzindo custos e melhorando o planeamento operacional.

Este projeto tem, portanto, relevância direta para o negócio da aviação comercial: a capacidade de identificar voos em risco antes da partida apoia a tomada de decisão sobre afetação de recursos, comunicação proativa com passageiros e otimização de operações em dias de maior risco — traduzindo técnicas de *machine learning* em valor operacional concreto.

---

## 2. Objetivo SMART

Desenvolver e avaliar dois modelos de classificação binária — um para prever o cancelamento de voo e outro para prever atraso significativo igual ou superior a 15 minutos — utilizando dados históricos de 1 041 151 voos comerciais norte-americanos de 2024, com *ROC-AUC* superior a 0.80 no modelo de cancelamentos e superior a 0.70 no modelo de atrasos, avaliados também por F1-*score* e *Avg Precision* em validação cruzada estratificada com cinco partições, até ao final da unidade curricular de Projeto em Ciência de Dados (maio de 2026).

---

## 3. Perguntas de Investigação

1. Quais as variáveis com maior poder preditivo na previsão de cancelamentos e atrasos, segundo a análise de importância *SHAP*?
2. É possível construir um classificador de cancelamentos com *ROC-AUC* superior a 0.80, num conjunto de dados com apenas 1.53% de casos positivos, sem acesso a dados meteorológicos em tempo real?
3. Variáveis temporais como o dia do mês e o mês do ano apresentam impacto relevante nas perturbações operacionais, traduzido em valores *SHAP* positivamente significativos para os modelos?
4. Aeroportos de origem com menor dimensão operacional apresentam taxas de cancelamento superiores às dos grandes *hubs*, e essa diferença é capturada pelo modelo como variável com importância elevada?
5. A engenharia de atributos — nomeadamente `is_weekend`, `is_long_flight` e `is_short_flight` — contribui para uma melhoria mensurável do F1-*score* face a um modelo treinado apenas com as variáveis originais?

---

## 4. Ferramentas e Bibliotecas

| Ferramenta / Biblioteca | Versão utilizada | Função no projeto |
|---|---|---|
| Python | 3.11 | Linguagem principal |
| pandas | 2.x | Manipulação e limpeza de dados |
| scikit-learn | 1.4.x | Pré-processamento, modelação, validação cruzada, `TunedThresholdClassifierCV` |
| XGBoost | 2.x | Algoritmo de *gradient boosting* para o modelo de atrasos |
| SHAP | 0.45.x | Interpretabilidade global e local dos modelos |
| Streamlit | 1.x | Desenvolvimento da aplicação web *FlightSense* |
| Plotly / Matplotlib / Seaborn | — | Visualização de dados na EDA e nos relatórios |
| Jupyter Notebook / Kaggle Code | — | Ambiente de desenvolvimento interativo |
| GitHub | — | Controlo de versões e repositório do projeto |

---

## 5. Metodologia de Gestão

- **Rodrigo Ramos:** Responsável pelo *setup* da infraestrutura (GitHub/Kaggle), engenharia e visualização de dados.
- **Bruno Almeida:** Responsável pela escrita de documentação técnica e modelação estatística.
- **Ferramentas de colaboração:** GitHub Projects para gestão de tarefas em *kanban* e reuniões regulares por Discord.

---

## 6. Análise de Viabilidade dos Dados

### Disponibilidade

O *dataset* **Flight Delay and Cancellation Data (1 Million+ 2024)** encontra-se disponível publicamente na plataforma [Kaggle](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data), com origem nos registos operacionais do *Bureau of Transportation Statistics* (BTS) do Departamento de Transportes dos EUA.

- Os dados foram descarregados e importados para o ambiente Kaggle Code.
- O repositório encontra-se estruturado com as pastas `data/raw/` (dataset original) e `data/processed/` (dados pós-tratamento).
- O *dataset* possui **1 048 575 linhas e 18 colunas**, correspondendo a registos individuais de voos comerciais realizados em 2024.

---

### Dicionário de Variáveis

O *dataset* contém **18 variáveis**, distribuídas da seguinte forma:

#### Variáveis Numéricas Inteiras

| Variável | Tipo | Gama de Valores | Descrição |
|---|---|---|---|
| `year` | Numérico inteiro | 2024 | Ano do voo |
| `month` | Numérico inteiro | 1–12 | Mês do voo |
| `day_of_month` | Numérico inteiro | 1–31 | Dia do mês |
| `day_of_week` | Numérico inteiro | 1–7 | Dia da semana (1 = segunda-feira) |
| `distance` | Numérico inteiro | 31–5 095 | Distância do voo em milhas |
| `cancelled` | Binário | {0, 1} | Indicador de cancelamento (0 = Não, 1 = Sim) — **variável-alvo 1** |

#### Variáveis Numéricas Reais (com valores nulos nos voos cancelados)

| Variável | Tipo | Gama de Valores | Descrição |
|---|---|---|---|
| `dep_time` | Numérico real | 0001–2359 | Hora real de partida (formato HHMM) |
| `taxi_out` | Numérico real | 1–250 | Minutos desde a saída do *gate* até descolagem |
| `wheels_off` | Numérico real | 0001–2359 | Hora de descolagem |
| `wheels_on` | Numérico real | 0001–2359 | Hora de aterragem |
| `taxi_in` | Numérico real | 1–200 | Minutos desde aterragem até ao *gate* |
| `air_time` | Numérico real | 15–700 | Tempo total de voo em minutos |
| `weather_delay` | Numérico real | 0–600 | Minutos de atraso por condições meteorológicas |
| `late_aircraft_delay` | Numérico real | 0–1 200 | Minutos de atraso por chegada tardia da aeronave |

#### Variáveis Categóricas

| Variável | Tipo | Descrição |
|---|---|---|
| `fl_date` | Categórico (data) | Data completa do voo (formato `YYYY-MM-DD`) |
| `origin` | Categórico nominal | Código IATA do aeroporto de origem (3 letras) |
| `origin_city_name` | Categórico nominal | Nome da cidade de origem |
| `origin_state_nm` | Categórico nominal | Estado norte-americano de origem |

#### Variável-Alvo Derivada

A segunda variável-alvo, `is_delayed`, não existe no *dataset* original e foi construída durante o pré-processamento com base em:

```
is_delayed = 1  se  weather_delay + late_aircraft_delay ≥ 15 minutos  (e voo não cancelado)
is_delayed = 0  caso contrário
```

Esta definição exclui voos cancelados e considera apenas atrasos com causa operacional identificada, resultando numa taxa de 8.52% de casos positivos.

---

### Qualidade Inicial dos Dados

A análise preliminar do *dataset* permitiu identificar os seguintes aspetos:

- **Valores nulos:** presentes em oito variáveis — `dep_time`, `taxi_out`, `wheels_off`, `wheels_on`, `taxi_in`, `air_time`, `weather_delay` e `late_aircraft_delay`. Os nulos em `dep_time` a `air_time` são estruturalmente coerentes: ocorrem nos voos cancelados, onde não existem dados de execução. Os nulos em `weather_delay` e `late_aircraft_delay` indicam ausência de atraso registado.
- **Duplicados:** identificados e removidos aproximadamente 7 400 registos duplicados.
- **Anomalias:** cerca de 23 registos com velocidades de cruzeiro fisicamente impossíveis foram removidos como *outliers*.
- **Desequilíbrio das classes:** a variável `cancelled` apresenta apenas 1.53% de casos positivos; `is_delayed` apresenta 8.52%. Este desequilíbrio exige estratégias específicas de modelação, como `class_weight='balanced'` e validação cruzada estratificada.
- **Variáveis temporais:** `fl_date`, `month`, `day_of_month` e `day_of_week` encontram-se em formatos `object` ou inteiro, exigindo conversão ou codificação adequada para modelação.
- **Exclusão por *data leakage*:** variáveis como `taxi_out`, `wheels_off`, `wheels_on`, `taxi_in`, `air_time`, `weather_delay` e `late_aircraft_delay` não estão disponíveis antes da partida do voo e foram excluídas dos modelos preditivos.

---

### Ética e Conformidade

O *dataset* é público, disponibilizado para fins académicos e de investigação na plataforma Kaggle, com origem em dados governamentais norte-americanos.

- Não contém dados pessoais identificáveis nem informação sensível relativa a passageiros.
- Contém exclusivamente dados operacionais de voos comerciais.
- Não existem implicações diretas relacionadas com o RGPD (Regulamento Geral sobre a Proteção de Dados), uma vez que os dados se encontram completamente anonimizados.

---

## 7. Cronograma

| Fase | Data Limite | Entregável |
|---|---|---|
| M1 — Iniciação | 24/02/2026 | Repositório estruturado e plano de projeto |
| M2 — Exploração | 24/03/2026 | *Notebook* de EDA e dados processados |
| M3 — Modelação | 28/04/2026 | Comparação de algoritmos e métricas de avaliação |
| M4 — Conclusões | 26/05/2026 | *Pitch* e relatório final |

---

## Referências

- **Nadeem, A. (2024).** *Flight Delay & Cancellation Data (1 Million+ 2024)*. Kaggle. Disponível em: [kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024)
- **Bureau of Transportation Statistics — BTS. (2024).** *Marketing Carrier On-Time Performance Data*. U.S. Department of Transportation. Disponível em: [transtats.bts.gov](https://www.transtats.bts.gov)
- **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

---

*Data de última atualização: maio de 2026*

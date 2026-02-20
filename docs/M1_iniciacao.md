# Milestone 1: Iniciação e Definição do Projeto
## 1. Descrição Detalhada do Problema
O transporte aéreo é um elemento essencial da mobilidade global, suportando atividades económicas, turismo e comércio internacional. Apesar do elevado nível de planeamento e tecnologia envolvido na aviação comercial, os atrasos e cancelamentos de voos continuam a ocorrer com frequência, afetando passageiros, companhias aéreas e aeroportos. Estas ocorrências geram impactos operacionais e financeiros significativos, tornando-se um problema relevante tanto do ponto de vista económico como logístico.

Este projeto baseia-se no dataset Flight Delay and Cancellation Data (1 Million+ 2024), disponibilizado no Kaggle, que contém mais de um milhão de registos de voos comerciais em companhias aéreas dos Estados Unidos realizados em 2024 . O conjunto de dados inclui informação como horários programados, aeroportos de origem e destino, distância do voo e indicadores de atraso e cancelamento.

O problema principal consiste em prever se um determinado voo irá sofrer atraso ou será cancelado, com base em dados históricos. Trata-se de um problema de classificação supervisionada, podendo também ser explorado como regressão no caso da previsão do número de minutos de atraso. A solução requer a análise e preparação dos dados, identificação de padrões temporais e operacionais e aplicação de algoritmos de aprendizagem automática adequados.

Este problema é relevante no momento atual pois existe a necessidade de tomar decisões baseadas em dados no setor da aviação. A capacidade de prever atrasos e cancelamentos permite melhorar o planeamento operacional, reduzir custos associados e aumentar a satisfação dos passageiros através de comunicação e gestão mais eficazes. Assim, este projeto demonstra como técnicas de ciência de dados podem contribuir para a otimização de operações num setor altamente dinâmico e competitivo.

## 2. Objetivos SMART
*Desenvolver um modelo preditivo capaz de antecipar atrasos em voos comerciais com base em dados históricos operacionais.*
* **S - Specific:** Prever se um voo irá sofrer atraso significativo (ex.: ≥15 minutos).
* **M - Measurable:** Avaliar o desempenho do modelo através de Accuracy, F1-score, ...
* **A - Achievable:** Utilização de dataset real com 1M+ de registos.
* **R - Relevant:** Aplicação direta em logística aérea e otimização operacional.
* **T - Time-bound:** Desenvolvimento do modelo e avaliação até ao final da unidade curricular.

## 3. Perguntas de Investigação
* 1. Quais os principais fatores que contribuem para o atraso de voos?
* 2. É possível prever com precisão se um voo irá atrasar antes da sua partida?
* 3. Existem padrões temporais (hora do dia, dia da semana, estação) associados a maiores probabilidades de atraso?
* 4. Certas companhias aéreas ou aeroportos apresentam maior probabilidade de atrasos?
* 5. Que variáveis têm maior influência na ocorrência de atrasos e de que forma contribuem para a sua previsão?

## 4. Metodologia de Gestão (PBL)
* **Rodrigo Ramos:** Responsável pelo setup da infraestrutura (GitHub/Kaggle), Engenharia e Visualização de Dados.
* **Bruno Almeida:** Responsável pela escrita de documentação técnica e modelação estatística.
* **Ferramentas de Colaboração:** GitHub Projects para KanBan e reuniões frequentes pelo Discord ou TeamSpeak.

## 5. Análise de Viabilidade dos Dados
### Disponibilidade
O dataset **Flight Delay and Cancellation Data (1 Million+ 2024)** encontra-se disponível publicamente na plataforma [Kaggle](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data).

- Os dados já foram descarregados e importados para o ambiente Kaggle Code.
- O repositório encontra-se estruturado com as pastas:
  - `data/raw/` — armazenamento do dataset original
  - `data/processed/` — dados após tratamento e transformação
- O dataset possui **1.048.575 linhas e 18 colunas**, correspondendo a registos individuais de voos comerciais realizados em 2024.
- Os dados encontram-se em formato **CSV**, sendo adequados para processamento em Python com `pandas`.

---

### Estrutura das Variáveis

Após a inspeção inicial (`df.info()`), o dataset contém **18 variáveis**, distribuídas da seguinte forma:

#### Variáveis Numéricas

| Variável | Tipo | Descrição |
|-----------|------|------------|
| `year` | int64 | Ano do voo |
| `month` | int64 | Mês do voo |
| `day_of_month` | int64 | Dia do mês |
| `day_of_week` | int64 | Dia da semana (1–7) |
| `dep_time` | float64 | Hora real de partida (formato HHMM, pode conter nulos) |
| `taxi_out` | float64 | Tempo (minutos) desde a partida do gate até descolagem |
| `wheels_off` | float64 | Hora em que o avião descolou |
| `wheels_on` | float64 | Hora em que o avião aterrou |
| `taxi_in` | float64 | Tempo (minutos) desde aterragem até ao gate |
| `cancelled` | int64 | Indicador binário de cancelamento (0 = Não, 1 = Sim) |
| `air_time` | float64 | Tempo total de voo em minutos |
| `distance` | int64 | Distância do voo (milhas) |
| `weather_delay` | int64 | Minutos de atraso atribuídos a condições meteorológicas |
| `late_aircraft_delay` | int64 | Minutos de atraso devido à chegada tardia da aeronave |

---

#### Variáveis Categóricas

| Variável | Tipo | Descrição |
|-----------|------|------------|
| `fl_date` | object | Data completa do voo (necessita conversão para datetime) |
| `origin` | object | Código do aeroporto de origem |
| `origin_city_name` | object | Nome da cidade de origem |
| `origin_state_nm` | object | Estado de origem |

---

### Qualidade Inicial dos Dados
A análise preliminar permitiu identificar os seguintes pontos:

- A estrutura do dataset é consistente e adequada para modelação supervisionada.
- Existem valores nulos em algumas colunas, nomeadamente em variáveis associadas a atrasos ou cancelamentos.
- As variáveis temporais encontram-se no formato `object`, sendo necessária conversão para `datetime`.
- Não foram identificadas colunas completamente vazias.
- Será necessário verificar:
  - Percentagem de valores nulos por variável;
  - Existência de registos duplicados;
  - Possíveis inconsistências (ex.: atrasos negativos);
  - Distribuição da variável alvo (equilíbrio entre classes).

Estas verificações serão aprofundadas na Milestone 2 (Exploração e Preparação dos Dados).

---

### Ética e Conformidade

O dataset é público e disponibilizado para fins académicos na plataforma Kaggle.

- Não contém dados pessoais identificáveis.
- Não inclui informação sensível relativa a passageiros.
- Contém exclusivamente dados operacionais de voos comerciais.

Assim:

- Não existem implicações diretas relacionadas com o RGPD (Regulamento Geral sobre a Proteção de Dados).
- Os dados encontram-se anonimizados.

---


## 6. Cronograma Interno
| Fase | Data Limite | Entregável Esperado |
| :--- | :--- | :--- |
| M1: Iniciação | 24/02 | Repositório estruturado e Plano de Projeto. |
| M2: Exploração | [Data] | Notebook de EDA e Dados Processados. |
| M3: Modelação | [Data] | Comparação de algoritmos e métricas. |
| M4: Finalização| [Data] | Pitch e Relatório Final. |
---
*Data de última atualização: 20/02/2026*

# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no
ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis,
deve consultar essa Milestone.

---

## 1. Análise Exploratória de Dados (EDA)

### 1.1. Distribuição das Variáveis-Alvo

Este projeto prevê **dois tipos de perturbação** em voos comerciais, correspondendo a dois problemas de classificação distintos:

| Tarefa | Variável-Alvo | Natureza |
|--------|---------------|----------|
| Previsão de cancelamentos | `cancelled` (0/1) | Classificação binária |
| Previsão de atrasos | `is_delayed` (0/1) | Classificação binária |

#### Variável `cancelled`

A variável `cancelled` indica se um voo foi cancelado (1) ou não (0). Trata-se de uma variável binária que apresenta um forte desequilíbrio entre classes, característico de situações em que se tenta identificar ocorrências pouco frequentes.

> **Factos importantes:**  
> A variável alvo encontra-se fortemente desequilibrada, com aproximadamente 98,47% de voos não cancelados e apenas 1,53% de voos cancelados.

**Breve Conclusão**  
A forte desproporção entre classes indica um problema de classificação desequilibrado, sendo necessário considerar métricas adequadas (como recall, precision ou F1-score) e técnicas específicas de modelação (class_weight, threshold tuning) na fase de modelação, uma vez que a accuracy, isoladamente, poderá conduzir a interpretações enganadoras.

<img width="1283" height="541" alt="image" src="https://github.com/user-attachments/assets/7a348050-37d6-47c0-bf21-9fad17757d96" />

*(Figura 1 — Distribuição e proporção da variável-alvo `cancelled`)*

---

#### Variável `is_delayed`

O dataset não contém diretamente uma variável de atraso à partida, pelo que foi definida uma nova variável-alvo `is_delayed`. O critério adotado é: um voo é considerado atrasado se `weather_delay + late_aircraft_delay ≥ 15 minutos` **e** não foi cancelado. Esta definição é conservadora — não capta atrasos por decisão da transportadora, NAS ou segurança, que não estão disponíveis neste dataset.

> **Factos importantes:**  
> A taxa de atraso situa-se em 8,52%, tornando este problema menos desequilibrado e mais tratável do que o de cancelamentos. Os dois problemas são maioritariamente mutuamente exclusivos: um voo cancelado não é considerado atrasado nesta definição.

**Breve Conclusão**  
A criação da variável `is_delayed` permite expandir o âmbito do projeto para um segundo problema de classificação com características distintas. O menor desequilíbrio face aos cancelamentos simplifica a modelação, embora a definição conservadora adotada implique que parte dos atrasos reais possa não estar capturada.

<img width="1818" height="595" alt="image" src="https://github.com/user-attachments/assets/69004e39-f63e-4cd7-a608-ac9756365562" />

*(Figura 2 — Visão geral das duas variáveis-alvo e resultado combinado do voo)*

---

### 1.2. Correlações Relevantes

Nesta fase foram analisadas relações entre variáveis explicativas e ambas as variáveis-alvo, com base em visualizações e estatística descritiva.

#### Atributo `origin` (Aeroporto) vs. Alvo

O volume de voos está concentrado nos 15 maiores aeroportos de origem. A análise segmentada mostra que a taxa de cancelamento e a taxa de atraso variam significativamente entre aeroportos, sugerindo que fatores operacionais associados à infraestrutura e ao volume de tráfego influenciam ambos os tipos de perturbação.

<img width="967" height="642" alt="image" src="https://github.com/user-attachments/assets/7ea75bb7-a7af-479c-898c-4d33f6fec92e" />

*(Figura 3 — Top 15 Aeroportos de Origem por Volume de Voos)*

<img width="1735" height="648" alt="image" src="https://github.com/user-attachments/assets/1ca6b3d3-a954-472a-a986-5f32e573957d" />

*(Figura 4 — Taxa de cancelamento e de atraso por aeroporto de origem)*

**Breve Conclusão**  
Os aeroportos de maior dimensão concentram a maioria das observações. A análise revela variações consideráveis nas taxas de cancelamento e de atraso entre aeroportos, o que indica que o aeroporto de origem possui capacidade discriminativa relevante para ambos os problemas de modelação.

---

#### Atributo `weather_delay` e `late_aircraft_delay` vs. Alvo

Estas variáveis representam informação posterior ao voo — apenas existem se o voo ocorreu. Por este motivo, não são adequadas como features preditivas, constituindo antes a base da definição da variável-alvo `is_delayed`. A sua inclusão direta como atributos preditivos introduziria fuga de informação (*data leakage*).

<img width="1522" height="487" alt="image" src="https://github.com/user-attachments/assets/2869e26f-1776-470c-ac39-afe7119ee5b7" />

*(Figura 5 — Caracterização dos atrasos: distribuição e causa dominante)*

**Breve Conclusão**  
As variáveis `weather_delay` e `late_aircraft_delay` não apresentam utilidade como atributos preditivos, uma vez que constituem variáveis de natureza "pós-evento". A sua utilização directa introduziria *data leakage*, comprometendo a validade dos modelos. Foram, por isso, utilizadas exclusivamente na definição de `is_delayed` e subsequentemente excluídas do conjunto de modelação.

---

#### Atributo `distance` vs. Alvo

A análise da variável distância foi feita com recurso a histograma de distribuição geral e boxplots segmentados por cancelamento e por atraso.

<img width="1742" height="541" alt="image" src="https://github.com/user-attachments/assets/de6dc04b-075b-4aeb-8bef-b20cded8a30f" />

*(Figura 6 — Análise da variável distância face às duas variáveis-alvo)*

**Breve Conclusão**  
A análise da relação entre a distância e ambas as variáveis-alvo evidencia uma sobreposição significativa entre voos cancelados e não cancelados, e entre voos atrasados e não atrasados, não sendo possível identificar um padrão claro de separação entre classes com base nesta variável isolada. Conclui-se que a distância apresenta capacidade discriminativa limitada de forma autónoma, sendo necessário combiná-la com outras variáveis para melhorar o desempenho dos modelos.

---

#### Atributo `month` e `day_of_week` vs. Alvo

A análise temporal revelou variações nas taxas de cancelamento e de atraso tanto entre os meses disponíveis no dataset como entre os diferentes dias da semana.

<img width="1192" height="532" alt="image" src="https://github.com/user-attachments/assets/256097ce-b10b-4606-9483-3b71453e76b7" />

*(Figura 7 — Taxa de cancelamento e de atraso por mês)*

<img width="1302" height="532" alt="image" src="https://github.com/user-attachments/assets/7c0630f0-6106-4be8-83b4-72b2817efc71" />

*(Figura 8 — Taxa de cancelamento e de atraso por dia da semana)*

**Breve Conclusão**  
Verificam-se diferenças nas taxas de cancelamento e de atraso ao longo dos meses e dos dias da semana, sugerindo a existência de padrões temporais com capacidade preditiva. No entanto, dado que o conjunto de dados pode não incluir a totalidade dos meses do ano, a variação mensal deve ser interpretada com cautela, não sendo possível, nesta fase, identificar padrões sazonais completos.

---

## 2. Qualidade dos Dados e Limpeza

### 2.1. Tratamento de Dados em Falta (Missing Data)

**Colunas afetadas:**  
`air_time`, `wheels_on`, `taxi_in`, `taxi_out`, `wheels_off`, `dep_time`

**Estratégia adotada:** abordagem orientada pela interpretação contextual dos dados, tendo em conta o enquadramento e o significado das variáveis analisadas.

1. **Identificação da causa:**  
Os valores nulos não representam erros, mas sim consequências de voos cancelados (por exemplo, `dep_time` não existe quando o voo não ocorre). A relação entre nulidade e cancelamento foi confirmada analiticamente: praticamente 100% dos registos com `air_time`, `wheels_on` ou `dep_time` em falta correspondem a voos cancelados.

2. **Segmentação:**  
Para efeitos de análise exploratória, foram consideradas separadamente as observações correspondentes a voos operados e a voos cancelados, permitindo avaliar o comportamento das variáveis operacionais apenas nos casos em que estas estão efetivamente disponíveis.

3. **Justificação:**  
Os valores nulos foram mantidos no conjunto de dados, uma vez que, no contexto das variáveis operacionais, estão associados à não realização do voo. Assim, em vez de serem tratados como falhas, estes valores foram interpretados como informação estruturalmente coerente com o problema em análise.

---

### 2.2. Remoção de Registos Duplicados

- Foram identificados e removidos aproximadamente 7 400 registos duplicados (~0,7% do dataset).

**Justificação:**  
Cada voo representa um evento único. A presença de registos duplicados indica problemas de integridade dos dados e pode introduzir enviesamento na aprendizagem do modelo, aumentando o risco de overfitting.

---

### 2.3. Outliers e Inconsistências

#### Erros físicos

- Foram identificados aproximadamente 23 registos com velocidades fisicamente impossíveis (velocidades supersónicas ou abaixo do mínimo operacional).
- Estes registos foram removidos.

**Justificação:**  
Tratam-se de erros de medição ou registo, não representando fenómenos que efetivamente aconteceram.

---

#### Outliers Operacionais

- Foram identificados valores extremos em variáveis como `late_aircraft_delay`, `taxi_in`, `taxi_out` e `weather_delay`, quantificados por análise IQR (1,5×).
- Estas variáveis não foram incluídas no dataset final de modelação.

<img width="1632" height="864" alt="image" src="https://github.com/user-attachments/assets/32478453-da67-4b7e-9cc2-9ee00c1b8463" />

*(Figura 9 — Análise de outliers por variável operacional)*

**Justificação:**  
Embora representem situações reais e informativas, estas variáveis apenas estão disponíveis durante ou após a realização do voo, não sendo úteis no momento da previsão. A sua inclusão como atributos preditivos introduziria fuga de informação (*data leakage*), comprometendo a veracidade e consistência dos resultados obtidos pelo modelo.

---

### 2.4. Sumário da Limpeza

| Operação | Registos afetados |
|----------|-------------------|
| Remoção de duplicados | ~7 400 |
| Remoção de erros físicos de velocidade | ~23 |
| Valores em falta | Mantidos — estruturalmente coerentes com cancelamentos |

---

## 3. Engenharia de Atributos (Feature Engineering)

### 3.1. Transformações Realizadas

- **Encoding:** Aplicação de One-Hot Encoding (técnica que transforma uma variável categórica em várias variáveis binárias (0 ou 1)), na variável `origin`. Foi utilizado `drop_first=True` para evitar multicolinearidade perfeita.
- **Escalonamento:** Não foi aplicado `StandardScaler` nem qualquer normalização. Os modelos utilizados na fase de modelação (HistGradient Boosting e XGBoost) são baseados em árvores de decisão — invariantes a transformações monotónicas como a normalização — pelo que o escalonamento não produz qualquer benefício e foi deliberadamente omitido.
- **Remoção de multicolinearidade:** Colunas com correlação absoluta superior a 0,85 entre features foram automaticamente identificadas e removidas. Nenhum par atingiu este limiar no dataset final.

---

### 3.2. Criação de Novos Atributos

Foram criadas novas variáveis com potencial relevância para ambos os problemas de modelação:

- **`is_long_flight`**  
  Indicador binário que identifica voos de longa distância, assumindo o valor 1 para voos com distância superior a 1 500 milhas e 0 caso contrário.

- **`is_short_flight`**  
  Indicador binário para voos de curta distância, assumindo o valor 1 para voos com distância inferior a 300 milhas. Complementa `is_long_flight` ao distinguir explicitamente a categoria de voos mais curtos.

- **`is_weekend`**  
  Indicador binário que assinala se o voo ocorre ao fim de semana (Sábado ou Domingo), capturando padrões operacionais associados a dias de maior procura de lazer.

- **`flight_period`**  
  Segmentação do dia em quatro períodos com base na hora de partida: madrugada, manhã, tarde e noite. Esta variável é criada e analisada na fase de EDA mas é excluída do conjunto de modelação final, uma vez que a hora de partida real (`dep_time`) não está disponível antes da partida do voo — o que tornaria a sua utilização preditiva inválida.

<img width="1742" height="1080" alt="image" src="https://github.com/user-attachments/assets/dc1c8de4-aed8-4875-b5b6-d9b97aadb122" />

*(Figura 10 — Impacto das novas variáveis nas taxas de cancelamento e de atraso)*

Estas variáveis permitem capturar padrões não evidentes nas variáveis originais.

---

## 4. Seleção de Atributos

Foi construída uma versão do dataset focada em variáveis disponíveis **antes da partida do voo**, evitando fuga de informação (*data leakage*). O conjunto final contém **ambas as variáveis-alvo** (`cancelled` e `is_delayed`), permitindo que a fase de modelação utilize qualquer uma delas sem necessidade de reprocessamento.

**Variáveis removidas:**
- Operacionais (pós-evento): `dep_time`, `taxi_out`, `wheels_off`, `wheels_on`, `taxi_in`, `air_time`
- Definidoras do alvo: `weather_delay`, `late_aircraft_delay`, `total_delay`
- Redundantes ou de identificação: `fl_date`, `year`, `origin_city_name`, `origin_state_nm`

**Justificação:**  
Estas variáveis não estão disponíveis no momento da previsão, são utilizadas na construção da variável-alvo `is_delayed` (não podendo ser também features), ou introduzem redundância sem acrescentar capacidade preditiva.

---

## 5. Dicionário de Dados Final

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `month` | Numérico inteiro | Mês do voo (1–12) |
| `day_of_month` | Numérico inteiro | Dia do mês (1–31) |
| `day_of_week` | Numérico inteiro | Dia da semana (1–7) |
| `origin_*` | Binário (*one-hot*) | Aeroporto de origem codificado (uma coluna por aeroporto) |
| `distance` | Numérico real | Distância do voo em milhas (31–5 095) |
| `is_long_flight` | Binário | Indicador de voo longo (> 1 500 milhas): {0, 1} |
| `is_short_flight` | Binário | Indicador de voo curto (< 300 milhas): {0, 1} |
| `is_weekend` | Binário | Indicador de fim de semana: {0, 1} |
| `cancelled` | Binário | Variável-alvo 1 — cancelamento: {0, 1} |
| `is_delayed` | Binário | Variável-alvo 2 — atraso ≥ 15 min (voo não cancelado): {0, 1} |

> **Nota:** A variável `flight_period` foi criada e analisada na EDA mas não integra o conjunto de modelação final — a hora de partida real (`dep_time`) não está disponível antes do voo, tornando a sua utilização preditiva inválida.

---

## 6. Conclusões da Fase de Exploração

A análise exploratória permitiu compreender melhor a estrutura, qualidade e limitações do dataset, destacando-se os seguintes aspetos:

- **Dois problemas de classificação** com características distintas: o cancelamento apresenta forte desequilíbrio (~2,2%), exigindo atenção especial na modelação; o atraso é menos desequilibrado (10–20%) e potencialmente mais tratável.
- **Forte desequilíbrio da variável `cancelled`**, típico de problemas de eventos raros, que exige o uso de métricas adequadas como recall, precision ou F1-score, em detrimento da accuracy isolada.
- **Existência de variáveis com risco de *data leakage***, cuja remoção foi essencial para garantir a validade dos modelos — em particular as variáveis operacionais e as que definem `is_delayed`.
- **Presença de outliers operacionais** que, embora extremos, representam situações reais; a sua exclusão do conjunto de modelação deve-se ao carácter pós-evento e não a serem inválidos.
- **Capacidade discriminativa limitada de variáveis isoladas** como `distance`, sendo necessária a combinação de múltiplos atributos para distinguir adequadamente as classes.
- **Padrões temporais identificados** tanto ao nível mensal como semanal para ambas as variáveis-alvo, sugerindo que a dimensão temporal possui relevância preditiva.

Adicionalmente, a criação de novas variáveis (`is_long_flight`, `is_short_flight`, `is_weekend`) permitiu capturar padrões não evidentes nos atributos originais. A variável `flight_period` foi criada e analisada na EDA mas excluída do conjunto de modelação, por depender da hora de partida real — não disponível antes do voo. De forma global, os dados, após a limpeza e preparação, apresentam qualidade suficiente para avançar para a fase de modelação.

---

*Data de última atualização: 02/05/2026*

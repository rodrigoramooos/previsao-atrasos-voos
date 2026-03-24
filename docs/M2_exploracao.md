# Milestone 2: Análise Exploratória e Engenharia de Atributos

> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no
ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis,
deve consultar essa Milestone.

---

## 1. Análise Exploratória de Dados (EDA)

### 1.1. Distribuição da Variável Alvo

A variável alvo considerada neste projeto é `cancelled`, que indica se um voo foi cancelado (1) ou não (0). Trata-se de uma variável binária que apresenta um forte desequilíbrio entre classes, característico de situações em que se tenta identificar ocorrências pouco frequentes.

> **Factos importantes:**  
A variável alvo encontra-se fortemente desequilibrada, com 1 025 269 voos não cancelados (97,78%) e apenas 23 306 voos cancelados (2,22%).

**Breve Conclusão**  
A forte desproporção entre classes indica um problema de classificação, sendo necessário considerar métricas adequadas (como recall, precision ou F1-score) na fase de modelação, uma vez que a accuracy, isoladamente, poderá conduzir a interpretações enganadoras.

---

### 1.2. Correlações Relevantes

Nesta fase foram analisadas relações entre variáveis explicativas e a variável alvo, com base em visualizações e estatística descritiva.

#### Atributo `origin` (Aeroporto) vs. Alvo

Verificou-se que o número de voos está fortemente concentrado em grandes aeroportos, como ATL e DFW, o que indica que a infraestrutura aeroportuária e o volume de tráfego podem influenciar a probabilidade de ocorrência de certas perturbações.



<img width="867" height="514" alt="image" src="https://github.com/user-attachments/assets/6790fd62-658b-43fb-b995-b41e2becf9bb" />


                                            (Figura 1 - Gráfico "Top 10 Aeroportos de Origem")

**Breve Conclusão**  
Os aeroportos de maior dimensão concentram a maioria das observações, o que sugere que fatores operacionais associados a grandes aeroportos podem influenciar a ocorrência de cancelamentos.

---

#### Atributo `weather_delay` vs. Alvo

Observou-se que a maioria dos voos apresenta `weather_delay = 0` e que os cancelamentos se concentram maioritariamente neste grupo.
Isto indica que esta variável representa informação posterior ao voo (ou seja, apenas existe se o voo ocorreu), não sendo adequada para prever cancelamentos.



<img width="679" height="447" alt="image" src="https://github.com/user-attachments/assets/0dedd03e-ca4a-421d-a269-ea0706947a3f" />


                                            (Figura 2 - Countplot "Cancelamento vs Weather Delay")

**Breve Conclusão**  
A variável `weather_delay` não apresenta utilidade para o problema em análise, uma vez que constitui uma variável de natureza "pós-evento", ou seja, que apenas apresenta informação posterior ao voo, introduzindo risco de fuga de informação (*data leakage*) caso fosse utilizada na modelação.

---

#### Atributo `distance` vs. Alvo

A análise da variável distância revelou uma forte sobreposição entre as distribuições de voos cancelados e não cancelados.



<img width="836" height="504" alt="image" src="https://github.com/user-attachments/assets/caca6107-599e-4a0f-af55-41effbdc5770" />


                                            (Figura 3 - Strip Plot "Distância vs Cancelamento")

**Breve Conclusão**  
A análise da relação entre a distância e o cancelamento evidencia uma sobreposição significativa entre voos cancelados e não cancelados, não sendo possível identificar um padrão claro de separação entre as classes com base nesta variável. Verifica-se que os voos não cancelados apresentam maior dispersão ao longo das diferentes distâncias, enquanto os voos cancelados seguem uma distribuição semelhante, ainda que com menor frequência. Assim, conclui-se que a distância, de forma isolada, apresenta uma capacidade discriminativa limitada, sendo necessário combiná-la com outras variáveis para melhorar o desempenho dos modelos.

---

#### Atributo `month` vs. Alvo

A análise temporal revelou variações na taxa de cancelamento entre os meses disponíveis no dataset.



<img width="833" height="456" alt="image" src="https://github.com/user-attachments/assets/7dd8d6b3-c8ef-4117-b8d6-b29d0f5ddece" />


                                            (Figura 4 - Gráfico "Taxa de Cancelamento por Mês")

**Breve Conclusão**  
Verifica-se uma diferença significativa na taxa de cancelamento entre os meses analisados, sendo esta consideravelmente mais elevada no mês 1 do que no mês 2. No entanto, dado que o conjunto de dados não inclui a totalidade dos meses do ano, esta variação deve ser interpretada com cautela, não sendo possível, nesta fase, identificar padrões sazonais consistentes.

---

## 2. Qualidade dos Dados e Limpeza

### 2.1. Tratamento de Dados em Falta (Missing Data)

**Colunas afetadas:**  
`air_time`, `wheels_on`, `taxi_in`, `taxi_out`, `wheels_off`, `dep_time`

**Estratégia adotada:** abordagem orientada pela interpretação contextual dos dados, tendo em conta o enquadramento e o significado das variáveis analisadas.

1. **Identificação da causa:**  
Os valores nulos não representam erros, mas sim consequências de voos cancelados (por exemplo, `dep_time` não existe quando o voo não ocorre).

2. **Segmentação:**  
Para efeitos de análise exploratória, foram consideradas separadamente as observações correspondentes a voos operados e a voos cancelados, permitindo avaliar o comportamento das variáveis operacionais apenas nos casos em que estas estão efetivamente disponíveis.

3. **Justificação:**  
Os valores nulos foram mantidos no conjunto de dados, uma vez que, no contexto das variáveis operacionais, estão associados à não realização do voo. Assim, em vez de serem tratados como falhas a apontar, estes valores foram interpretados como informação relevante para o problema em análise.

---

### 2.2. Remoção de Registos Duplicados

- Foram identificados 7 401 registos duplicados (~0,7% do dataset).
- Foram removidos todos os valores duplicados na totalidade.

**Justificação:**  
Cada voo representa um evento único. A presença de registos duplicados indica problemas de integridade dos dados e pode vir a introduzir "desigualdade" na aprendizagem do modelo, aumentando o risco de overfitting.

---

### 2.3. Outliers e Inconsistências

#### Erros físicos

- Foram identificados 23 registos com velocidades impossíveis (ex: velocidades supersónicas).
- Estes registos foram removidos.

**Justificação:**  
Tratam-se de erros de medição ou registo, não representando fenómenos que efetivamente aconteceram.

---

#### Outliers Operacionais

- Foram identificados valores extremos em variáveis como `late_aircraft_delay`, `taxi_in` e `taxi_out`.
- Estas variáveis não foram incluídas no dataset final de modelação.

**Justificação:**  
Embora representem situações reais e informativas, estas variáveis apenas estão disponíveis durante ou após a realização do voo, não sendo uteis no momento da previsão. A sua inclusão traria fuga de informação (*data leakage*), comprometendo a veracidade e a consistência dos resultados obtidos pelo modelo.

---

## 3. Engenharia de Atributos (Feature Engineering)

### 3.1. Transformações Realizadas

- **Encoding:** Aplicação de One-Hot Encoding (técnica que transforma uma variável categórica em várias variáveis binárias (0 ou 1)), por exemplo na variável `origin`.
- **Normalização:** Aplicação de normalização nas variáveis numéricas.

---

### 3.2. Criação de Novos Atributos

Foram criadas novas variáveis com potencial relevância para a modelação:

- **`is_long_flight`**  
  Indicador binário que identifica voos de longa distância, assumindo o valor 1 para voos com distância superior a 1500 km e 0 caso contrário.  
  O valor 1500 foi definido com base na distribuição da variável distância, permitindo distinguir de forma simples voos de curta e longa duração no contexto do dataset.

- **`flight_period`**  
  Segmentação do dia em períodos (madrugada, manhã, tarde, noite).

Estas variáveis permitem capturar padrões não evidentes nas variáveis originais.

---

## 4. Seleção de Atributos

Foi feita uma "versão" do dataset focada em variáveis disponíveis antes da partida do voo, evitando fuga de informação (*data leakage*).

**Variáveis removidas:**
- Operacionais: `dep_time`, `taxi_out`, `wheels_off`, etc.
- Pós-evento: `weather_delay`, `late_aircraft_delay`
- Redundantes: `fl_date`, `year`, `origin_city_name`

**Justificação:**  
Estas variáveis não estão disponíveis no momento da previsão ou introduzem redundância.

---

## 5. Dicionário de Dados Final

| Atributo | Tipo | Descrição |
|----------|------|----------|
| `month` | int64 | Mês do voo |
| `day_of_month` | int64 | Dia do mês |
| `day_of_week` | int64 | Dia da semana |
| `origin` | object | Aeroporto de origem |
| `distance` | float64 | Distância do voo |
| `is_long_flight` | int64 | Indicador de voo longo |
| `cancelled` | int64 | Variável alvo |

---

## 6. Conclusões da Fase de Exploração

A análise exploratória permitiu compreender melhor a estrutura, qualidade e limitações do dataset, destacando-se os seguintes aspetos:

- Forte desíquilibrio da variável alvo, típico de problemas de eventos raros, exigindo especial atenção na escolha das métricas de avaliação a utilizar
- Existência de variáveis com risco de *data leakage*, cuja remoção foi essencial para garantir a "veracidade" dos modelos  
- Presença de outliers operacionais que, embora extremos, representam situações reais e relevantes para o problema  
- Necessidade de criação de novas variáveis, uma vez que algumas das variáveis originais, por si só, não permitem distinguir de forma evidente, voos cancelados e não cancelados

Adicionalmente, verificou-se que nem todas as variáveis disponíveis contribuem de forma direta para a previsão do cancelamento, sendo necessário um processo cauteloso de seleção de atributos. Observou-se também que algumas relações podem ser influenciadas por limitações do próprio dataset, como a cobertura temporal incompleta. De forma global, os dados, após a limpeza e preparação, apresentam qualidade suficiente para avançar para a fase de modelação.

---

*Data de última atualização: 24/03/2026*

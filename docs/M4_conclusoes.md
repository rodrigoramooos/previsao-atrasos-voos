# Relatório de Conclusão e Entrega de Valor — Milestone 4

**Projeto:** Previsão de Atrasos e Cancelamentos de Voos Comerciais  
**Grupo:** 1 — Rodrigo Ramos (a2023137922) · Bruno Almeida (a2023143583)  

---

## 1. Síntese de Resultados e Impacto

### O Problema Resolvido

Na Milestone 1, definimos dois objetivos concretos: prever se um voo seria **cancelado** e se sofreria um **atraso significativo (≥ 15 minutos)**, utilizando apenas informação disponível *antes da partida* — sem acesso a dados meteorológicos em tempo real ou a causas de atraso registadas durante o voo.

Esses objetivos foram alcançados. Desenvolvemos dois modelos independentes, ambos baseados em **HistGradientBoostingClassifier**, treinados sobre mais de **1 milhão de registos reais de voos norte-americanos (2024)**.

---

### Interpretação dos Resultados em Linguagem Simples

**Cancelamentos:**  
O modelo analisa o *quando* e o *de onde* de um voo e estima a probabilidade de este não chegar a operar. Com um limiar de decisão ajustado (0.806), o modelo identifica corretamente **~1 em cada 3 voos que seriam cancelados** — uma capacidade real de antecipação num cenário onde os cancelamentos representam apenas 1.53% do total de voos e o desequilíbrio entre classes é de **64:1**. Por cada cancelamento corretamente detetado, o sistema emite em média **8 falsos alarmes** — um rácio elevado, mas justificável pelo custo assimétrico entre um alarme desnecessário e um cancelamento não antecipado.

**Atrasos:**  
O modelo de atrasos opera num contexto estruturalmente mais favorável (desequilíbrio **11:1**), o que se reflete no desempenho. Identifica cerca de **4 em cada 10 voos que chegarão com atraso**, com um rácio de **3.5 falsos alarmes por deteção correta** — significativamente mais eficiente do que o modelo de cancelamentos. O espaço de ajuste do threshold é real: ao contrário dos cancelamentos, é possível subir ou baixar o ponto de operação sem colapso imediato das métricas.

---

### Desempenho Final dos Modelos

| | Cancelamentos | Atrasos |
|---|---|---|
| **Algoritmo** | HistGradient Boosting | HistGradient Boosting |
| **Desequilíbrio de classes** | 64:1 | 11:1 |
| **Threshold otimizado** | 0.806 | 0.606 |
| **F1-score** | 0.166 | 0.291 |
| **Recall** | 0.327 | 0.416 |
| **Precision** | 0.105 | 0.227 |
| **ROC-AUC** | 0.871 | 0.723 |
| **Avg Precision (PR-AUC)** | 0.100 | 0.226 |
| **TP (conjunto de teste)** | 1 040 | 7 380 |
| **FN (conjunto de teste)** | 2 141 | 10 358 |
| **FP (conjunto de teste)** | 8 340 | 25 554 |
| **FP/TP** | 8.0 | 3.5 |

> **Nota sobre o ROC-AUC:** O valor de 0.871 para cancelamentos parece superior ao de atrasos (0.723), mas esta inversão é enganosa. O AUC é inflacionado pelos ~197 000 "negativos". A **curva Precision-Recall (PR-AUC)** é a leitura "honesta" em contextos de desequilíbrio severo — e aí a vantagem de `is_delayed` (0.226 vs 0.100) é clara.

---

### Valor para o Negócio

| Beneficiário | Valor Gerado |
|---|---|
| **Companhias aéreas** | Antecipação de cancelamentos permite realocar tripulações e aeronaves com mais tempo, reduzindo custos de last-minute |
| **Aeroportos** | Identificação de dias e rotas de maior risco apoia o planeamento de pessoal e gates; a lista de aeroportos críticos (RSW, DEN, MCI, TTN, ANC) fornece alvos concretos de melhoria operacional |
| **Passageiros** | Notificações preventivas melhoram a experiência e reduzem situações de surpresa; a aplicação FlightSense democratiza o acesso a estas previsões |
| **Gestão operacional** | Os padrões sistemáticos por aeroporto — hubs de tráfego intenso (ORD, DFW, ATL) para atrasos; aeroportos de clima severo (BUF, MDW) para cancelamentos — permitem intervenções estruturais diferenciadas |

O principal benefício não está na precisão absoluta dos modelos, mas na **redução de falsos negativos**: cada cancelamento ou atraso que o modelo deteta antecipadamente, e que de outra forma seria ignorado, tem valor operacional concreto. Um falso alarme tem custo operacional marginal; um cancelamento não antecipado tem impacto direto em tripulações, gates e passageiros.

---

## 2. Interpretabilidade dos Modelos

A interpretabilidade foi garantida através de duas técnicas complementares, desenvolvidas no notebook de interpretação (3.0):

### 2.1 Permutation Importance

A importância por permutação foi calculada com **5 repetições** sobre uma **amostra estratificada de 4 000 observações** do conjunto de teste, usando a Average Precision como métrica de referência. Este método mede a queda no desempenho quando cada feature é aleatoriamente embaralhada — valores elevados indicam features das quais o modelo depende genuinamente, com barras de erro estreitas a confirmar estabilidade do sinal.

**Top features — `cancelled`:**  
`day_of_month` e `month` dominam com margem clara — o cancelamento é essencialmente um fenómeno temporal. `distance` e `day_of_week` surgem num segundo nível com sinal estável. `origin_BUF` (Buffalo) e `origin_MDW` (Chicago Midway) aparecem com importância baixa mas consistente, associados a cancelamentos de inverno no nordeste dos EUA.

**Top features — `is_delayed`:**  
Perfil semelhante no topo, mas `day_of_week` sobe para 3.º lugar — os atrasos têm padrão semanal mais marcado. Os aeroportos de destaque são grandes hubs: `origin_ORD` (Chicago O'Hare), `origin_DFW` (Dallas/Fort Worth), `origin_ATL` (Atlanta). Contrasta com `cancelled`, onde predominam aeroportos de clima severo — a divergência reflete diretamente as causas subjacentes de cada fenómeno: atrasos propagam-se em cadeia nos hubs; cancelamentos concentram-se por condições meteorológicas locais.

### 2.2 Sobreposição entre Modelos

A análise comparativa dos top-10 features de cada target identificou **6 features partilhadas**: `day_of_month`, `day_of_week`, `distance`, `month`, `origin_MIA` e `origin_ORD` — base preditiva comum a ambos os fenómenos.

As features exclusivas são analiticamente informativas:
- **Exclusivas de `cancelled`:** BUF, MDW, CLT, BOS — aeroportos em regiões com invernos severos
- **Exclusivas de `is_delayed`:** ATL, DCA, DEN, DFW — grandes hubs de tráfego intenso onde os atrasos se propagam sistemicamente

Esta divergência não é um artefacto do modelo — é um reflexo fiel da realidade operacional da aviação norte-americana.

---

## 3. Análise de Grupos de Risco

A análise por aeroporto cruzou a **taxa real de cancelamento/atraso** com o **Recall do modelo** por aeroporto, identificando quatro quadrantes operacionais. O quadrante mais crítico — **Alto Risco / Falha do Modelo** — agrupa aeroportos onde o risco é elevado mas o modelo não consegue cobrir adequadamente:

**Aeroportos críticos para `cancelled`:**

| Aeroporto | Taxa Real | Recall | Observação |
|---|---|---|---|
| RSW (Fort Myers) | 2.16% | 0.156 | Alto risco, cobertura insuficiente |
| DEN (Denver) | 1.95% | 0.207 | Prioritário pelo volume absoluto (174 eventos não detetados) |
| MCI (Kansas City) | — | 0.080 | Recall mais baixo de toda a amostra |

**Aeroportos críticos para `is_delayed`:**

| Aeroporto | Taxa Real | Recall | Observação |
|---|---|---|---|
| TTN (Trenton) | 20.7% | 0.235 | Taxa de atraso muito elevada, cobertura fraca |
| ANC (Anchorage) | — | 0.128 | Recall mais baixo do quadrante crítico, com volume elevado |
| BQN (Aguadilla) | — | — | Destaque no quadrante crítico |

Estes aeroportos devem ser alvo prioritário em qualquer iteração futura: são precisamente os contextos onde o modelo falha mais e onde o impacto operacional de um sistema de alerta seria maior.

---

## 4. Análise de Erros — O Que o Modelo Não Consegue Ver

### 4.1 Perfil dos Falsos Negativos

A análise comparou as probabilidades atribuídas pelo modelo aos **Verdadeiros Positivos** (eventos corretamente detetados) e **Falsos Negativos** (eventos reais não detetados):

| | Probabilidade mediana — TP | Probabilidade mediana — FN |
|---|---|---|
| `cancelled` | > 0.806 (threshold) | **0.659** |
| `is_delayed` | > 0.606 (threshold) | **0.455** |

Este resultado é analiticamente fundamental: os falsos negativos não são casos onde o modelo hesitou — são casos aos quais o modelo atribuiu **genuinamente baixa probabilidade de risco**. A distribuição dos FN concentra-se muito abaixo dos thresholds, sugerindo que estes eventos foram causados por fatores operacionais do dia que simplesmente não estão representados no dataset: condições meteorológicas pontuais, incidentes técnicos imprevistos, perturbações em cascata não captadas pelas features disponíveis.

**Conclusão:** O modelo não falha por incapacidade de discriminação — falha por ausência de informação. O teto de desempenho com as features atuais está estruturalmente limitado.

### 4.2 Rácio FP/TP — Eficiência Operacional

| Modelo | FP/TP | Interpretação |
|---|---|---|
| `cancelled` | **8.0** | Por cada cancelamento detetado, 8 falsos alarmes |
| `is_delayed` | **3.5** | Por cada atraso detetado, 3.5 falsos alarmes |

O modelo de atrasos é substancialmente mais eficiente do ponto de vista operacional. A diferença reflete o menor desequilíbrio de classes (11:1 vs 64:1) e a maior previsibilidade estrutural dos atrasos face aos cancelamentos.

---

## 5. Curvas de Desempenho e Análise de Threshold

### 5.1 Curvas PR e ROC

Na curva Precision-Recall, `is_delayed` (AP=0.226) ocupa área muito superior — a Precision de `cancelled` colapsa rapidamente ao aumentar o Recall, penalizando qualquer tentativa de detetar mais cancelamentos sem gerar volumes insustentáveis de falsos alarmes.

As curvas de **lift e gain** confirmam valor operacional real: nos primeiros decis (10-20% de maior probabilidade estimada), ambos os modelos são substancialmente mais eficientes do que monitorização uniforme — sinalizam uma proporção de eventos reais muito superior ao esperado por acaso.

### 5.2 Análise de Threshold — Margem de Ajuste Real

**`cancelled` (threshold 0.806) — margem mínima:**  
O pico de F1 é estreito. Baixar o threshold faz crescer o FP muito mais depressa do que o TP, tornando o sistema operacionalmente insustentável. A 0.40 detetam-se mais cancelamentos, mas o volume de falsos alarmes inviabiliza qualquer uso prático. Não existe margem real de ajuste.

**`is_delayed` (threshold 0.606) — flexibilidade real:**  
O pico de F1 é mais largo e o rácio FP/TP decresce mais suavemente ao subir o threshold. A 0.50 ainda existe Recall aceitável; a 0.70 a Precision sobe a custo do Recall. Existe espaço real de calibração operacional — um operador pode ajustar o ponto de trabalho conforme o custo relativo de falsos alarmes vs eventos não detetados no seu contexto específico.

---

## 6. Análise Crítica e Limitações

### Limitações dos Dados

- **Ausência de dados meteorológicos em tempo real.** Esta é a limitação mais crítica e explica diretamente o perfil dos falsos negativos. Tempestades, nevoeiro e neve são as principais causas reais de cancelamentos, mas não estão disponíveis antes do voo. O modelo compensa inferindo risco a partir de variáveis temporais e geográficas que *correlacionam* com mau tempo — mas não o observam diretamente.

- **Desequilíbrio extremo em `cancelled` (1.53%, 64:1).** Apesar das técnicas utilizadas (`class_weight='balanced'`, otimização de threshold por CV), o desequilíbrio limita estruturalmente o F1-score. Num conjunto de teste de 208 231 voos existem apenas 3 181 cancelamentos — qualquer erro sistemático tem impacto amplificado nas métricas.

- **Dataset circunscrito a 2024.** O modelo foi treinado e avaliado num único ano. Não é garantido que os padrões aprendidos se mantenham estáveis em anos com dinâmicas diferentes (greves, pandemias, eventos climáticos extremos sem precedente).

- **Granularidade temporal limitada.** O dataset contém o dia do mês e o mês, mas não a **hora de partida programada** — uma variável com impacto operacional significativo (voos matinais têm padrões distintos dos noturnos) e disponível na fonte BTS original.

### Limitações do Modelo

- **Correlação sem causalidade.** HistGradient Boosting identifica padrões históricos, não relações causais. O facto de `day_of_month` ser a feature mais importante não significa que o dia *cause* cancelamentos — ambos estão correlacionados com condições que o dataset não observa diretamente.

- **Threshold fixo pós-treino.** O limiar foi otimizado com `TunedThresholdClassifierCV` sobre dados históricos. Em produção, a distribuição real pode desviar-se e o threshold pode necessitar de recalibração periódica.

### Contextos de Falha

- **Eventos extraordinários** não contemplados no treino (greves, condições meteorológicas extremas sem precedente histórico) estão fora do alcance do modelo.
- **Aeroportos sub-representados** têm menos dados para suportar padrões locais; as previsões para rotas raras são estruturalmente menos fiáveis.
- **Alterações de última hora:** o modelo foi desenhado para previsão pré-partida com base nos atributos do voo programado. Se um voo sofrer *rescheduling*, esses dados não chegam ao modelo.

---

## 7. Considerações Éticas

### Privacidade

O dataset utilizado é completamente anónimo: não contém identificação de passageiros, tripulações ou funcionários. Todos os registos referem-se a operações de voo (companhia, aeroporto, data, métricas operacionais), sem qualquer dado pessoal. Não existem preocupações de privacidade associadas ao modelo desenvolvido.

### Transparência e Explicabilidade

Os modelos não operam como "caixas negras". Foram utilizadas duas técnicas complementares:

- **Permutation Importance** — ranking global de importância com intervalos de incerteza (5 repetições), permitindo distinguir sinal robusto de ruído
- **Análise SHAP** — explicação por observação individual, mostrando quanto cada variável contribuiu (positiva ou negativamente) para a probabilidade estimada num voo específico

Esta combinação garante que qualquer decisão tomada com base nos modelos pode ser justificada — requisito fundamental para deployment em sistemas de apoio à decisão.

---

## 8. Roadmap e Trabalhos Futuros

### Melhorias de Maior Impacto Esperado

1. **Integrar dados meteorológicos históricos e previsões em tempo real.**  
   Cruzar o dataset com fontes de previsões em tempo real por aeroporto seria a melhoria com maior impacto esperado — condições climáticas explicam diretamente grande parte dos cancelamentos que o modelo atual apenas infere indiretamente. A análise de falsos negativos (probabilidades medianas de 0.659 e 0.455) confirma que estes eventos têm causas não representadas nas features atuais.

2. **Adicionar hora de partida programada (`scheduled_departure_hour`).**  
   Disponível na fonte original mas ausente no dataset processado. Voos com horário de manhã acumulam atrasos do dia anterior; voos noturnos têm padrões distintos. Seria provavelmente uma das features mais informativas a adicionar.

3. **Histórico recente do avião (`tail_number`).**  
   Um atraso num voo anterior aumenta a probabilidade de atraso no seguinte. Esta dinâmica temporal não é capturada pelo modelo atual e é um dos principais determinantes de atrasos "em cascata".

### Escalabilidade e Deployment

4. **Interface web FlightSense (implementado).**  
   A aplicação está em produção em [previsao-cancelamento.streamlit.app](https://previsao-voos.streamlit.app/). Trabalho futuro: integração de explicabilidade SHAP por voo diretamente na interface, permitindo ao utilizador perceber *porquê* um voo específico foi sinalizado.

---

## 9. Conclusão

Este projeto demonstrou que é possível construir modelos preditivos com valor operacional real para o setor da aviação, utilizando exclusivamente informação disponível antes do voo — sem aceder a dados privilegiados ou em tempo real.

Os dois modelos desenvolvidos — **HistGradient Boosting para cancelamentos** e **HistGradient Boosting para atrasos** — não são perfeitos, e as suas limitações foram documentadas com "honestidade". O desequilíbrio extremo das classes, a ausência de dados meteorológicos e a circunscrição a um único ano são os principais condicionantes.

A análise de erros revelou um dado crítico que vai além das métricas: os falsos negativos têm probabilidades medianas de **0.659** e **0.455**, muito abaixo dos thresholds — o modelo não hesita nestes casos, simplesmente não tem informação suficiente para os detetar. **O problema não é o modelo: é o dataset.** Com as features disponíveis, o teto de desempenho está estruturalmente limitado, e os resultados obtidos estão próximos desse teto.

Ainda assim, os resultados constituem uma base sólida e auditável: os modelos são explicáveis via Permutation Importance, foram otimizados com validação cruzada estratificada, os thresholds foram calibrados para maximizar a utilidade prática (F1), e as curvas de lift e gain confirmam valor operacional real nos decis de maior risco — justificando a sua utilização num sistema de suporte à decisão.

O caminho para um sistema de produção está bem definido: **dados meteorológicos, hora de partida e histórico do avião** são as três melhorias com maior potencial de impacto. Com esses ingredientes, os modelos aqui desenvolvidos constituem um ponto de partida robusto para uma solução escalável de apoio à decisão no setor da aviação.

---

*Coimbra Business School | ISCAC — Licenciatura em Ciência de Dados para a Gestão*  
*Unidade Curricular: Projeto em Ciência de Dados — Docente: Dora Melo*

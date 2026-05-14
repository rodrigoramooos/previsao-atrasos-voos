# Relatório de Conclusão e Entrega de Valor — Milestone 4

**Projeto:** Previsão de Atrasos e Cancelamentos de Voos Comerciais  
**Grupo:** 1 — Rodrigo Ramos (a2023137922) · Bruno Almeida (a2023143583)  
**Data de Conclusão:** Maio de 2026  
**Versão:** v4.0 Final

---

## 1. Síntese de Resultados e Impacto

### O Problema Resolvido

Na Milestone 1, definimos dois objetivos concretos: prever se um voo seria **cancelado** e se sofreria um **atraso significativo (≥ 15 minutos)**, utilizando apenas informação disponível *antes da partida* — sem acesso a dados meteorológicos em tempo real ou a causas de atraso registadas durante o voo.

Esses objetivos foram alcançados. Desenvolvemos dois modelos independentes, treinados sobre mais de **1 milhão de registos reais de voos norte-americanos (2024)**, capazes de sinalizar preventivamente perturbações operacionais com base em padrões históricos.

---

### Interpretação dos Resultados em Linguagem Simples

**Cancelamentos:**  
O modelo de cancelamentos analisa o *quando* e o *de onde* de um voo e estima a probabilidade de este não chegar a operar. Com um limiar de decisão ajustado (0.806), o modelo consegue identificar corretamente **1 em cada 4 voos que seriam cancelados** — uma capacidade real de antecipação num cenário onde os cancelamentos representam apenas 1.53% do total de voos.

**Atrasos:**  
O modelo de atrasos identifica cerca de **4 em cada 10 voos que chegarão com atraso**, com um nível de precision razoável dado o desequilíbrio existente. Consegue distinguir dias e rotas de maior risco, oferecendo sinais operacionais com antecedência.

---

### Valor para o Negócio

| Beneficiário | Valor Gerado |
|---|---|
| **Companhias aéreas** | Antecipação de cancelamentos permite realocar tripulações e aeronaves com mais tempo, reduzindo custos de last-minute |
| **Aeroportos** | Identificação de dias/rotas de maior risco apoia o planeamento de pessoal e gates |
| **Passageiros** | Notificações preventivas melhoram a experiência e reduzem situações de surpresa |
| **Gestão operacional** | O modelo de atrasos sinaliza padrões sistemáticos por aeroporto de origem (ex: DFW como fator de risco) que permitem intervenções estruturais |

O principal benefício não está na precisão absoluta dos modelos, mas na **redução de falsos negativos**: cada cancelamento que o modelo deteta antecipadamente, e que de outra forma seria ignorado, tem valor operacional concreto. Um alerta preventivo adicional (falso positivo) tem custo muito inferior a um cancelamento não antecipado.

---

### Desempenho Final dos Modelos

| | Cancelamentos | Atrasos |
|---|---|---|
| **Algoritmo** | HistGradient Boosting | XGBoost |
| **F1-score** | 0.151 | 0.289 |
| **Recall** | 0.266 | 0.396 |
| **Precision** | 0.105 | 0.227 |
| **ROC-AUC** | 0.854 | 0.718 |
| **Avg Precision (PR-AUC)** | 0.087 | 0.225 |
| **Threshold otimizado** | 0.806 | 0.613 |

---

## 2. Análise Crítica e Limitações

### Limitações dos Dados

- **Ausência de dados meteorológicos em tempo real.** Esta é a limitação mais crítica. Fatores como tempestades, nevoeiro ou neve são as principais causas reais de cancelamentos e atrasos, mas não estão disponíveis no dataset antes do voo. O modelo compensa inferirei risco a partir de variáveis temporais e geográficas (mês, aeroporto) que *correlacionam* com mau tempo, mas não o observam diretamente.

- **Desequilíbrio extremo na classe `cancelled` (1.53%).** Apesar das técnicas utilizadas (`class_weight='balanced'`, otimização de *threshold* por CV), o desequilíbrio limita estruturalmente o F1-*score*. Num conjunto de teste de 208 231 voos, existem apenas 3 181 cancelamentos — qualquer erro sistemático tem impacto amplificado nas métricas.

- **Dataset circunscrito a 2024.** O modelo foi treinado e avaliado num único ano. Não é garantido que os padrões aprendidos (sazonalidade, aeroportos de risco) se mantenham estáveis em anos com dinâmicas diferentes (ex: anos com greves, pandemia, eventos extremos).

- **Granularidade temporal limitada.** O dataset contém o dia do mês e o mês, mas não a hora de partida programada — uma variável com impacto operacional significativo (voos noturnos têm padrões distintos dos matinais).

---

### Limitações do Modelo

- **Modelos baseados em árvores sem interpretação causal.** HistGradient Boosting e XGBoost identificam correlações históricas, mas não estabelecem causalidade. O facto de `day_of_month` ser a feature mais importante não significa que o dia *cause* cancelamentos — ambos estão correlacionados com condições que o dataset não observa diretamente.

- **Threshold fixo pós-treino.** O limiar de decisão foi otimizado com `TunedThresholdClassifierCV` sobre dados de validação históricos. Em produção, a distribuição real pode desviar-se e o threshold pode necessitar de recalibração periódica.

- **Ausência de ensemble entre modelos.** Cada target tem um único modelo vencedor. Um ensemble (stacking) de HistGBT + XGBoost poderia melhorar a robustez, especialmente nos casos de maior incerteza.

---

### Contextos de Falha

- **Eventos extraordinários não contemplados no treino:** greves, panes de sistema, pandemias ou condições meteorológicas extremas sem precedente histórico estão fora do alcance do modelo.

- **Aeroportos sub-representados:** aeroportos com poucos voos no dataset têm menos dados para suportar a aprendizagem de padrões locais. As previsões para rotas raras são menos fiáveis.

- **Voos de última hora / rescheduled:** o modelo foi desenhado para previsão pré-partida com base nos atributos do voo programado. Se um voo sofrer alterações de rota ou horário de última hora, esses dados não chegam ao modelo.

---

## 3. Considerações Éticas e de Viés

### Privacidade

O dataset utilizado é completamente anónimo: não contém identificação de passageiros, tripulações ou funcionários. Todos os registos referem-se a operações de voo (companhia, aeroporto, data, métricas operacionais), sem qualquer dado pessoal. Não existem preocupações de privacidade associadas ao modelo desenvolvido.

### Transparência e Explicabilidade

Os modelos não operam como "caixas negras". Foram utilizadas duas técnicas complementares de interpretabilidade:

- **Permutation Importance** — identifica, para cada modelo, quais as features que mais degradam o desempenho quando embaralhadas, fornecendo uma ranking global de importância.
- **SHAP (SHapley Additive exPlanations)** — explica cada previsão individual, mostrando quanto cada variável contribuiu (positiva ou negativamente) para a probabilidade estimada. Permite dizer, para um voo específico, *porquê* o modelo o sinaliza como de risco.

Esta combinação garante que qualquer decisão tomada com base nos modelos pode ser justificada e auditada.

### Viés Geográfico e Sazonal

As features de aeroporto de origem (variáveis one-hot) introduzem um viés geográfico intrínseco: aeroportos em regiões com inverno rigoroso (ex: Buffalo — BUF, Chicago Midway — MDW) surgem associados a maior risco, refletindo a realidade histórica mas podendo ser percebidos como discriminação geográfica. Este efeito é transparente no SHAP e resulta de padrões reais nos dados — não de uma escolha arbitrária do modelo.

---

## 4. Roadmap e Trabalhos Futuros

### Melhorias Técnicas

1. **Integrar dados meteorológicos históricos e previsões em tempo real.**  
   Cruzar o dataset com fontes como NOAA ou OpenWeatherMap por aeroporto e data seria a melhoria com maior impacto esperado. Condições climáticas explicam diretamente grande parte dos cancelamentos que o modelo atual apenas infere indiretamente.

2. **Testar técnicas de reamostragem (SMOTE, ADASYN).**  
   O desequilíbrio da classe `cancelled` (1.53%) foi tratado com `class_weight` e otimização de threshold, mas reamostragem sintética poderia ajudar o modelo a aprender melhores fronteiras de decisão para os casos positivos raros.

3. **Ensemble stacking (HistGBT + XGBoost).**  
   Os dois algoritmos têm desempenhos muito próximos em ambos os targets. Um meta-modelo de stacking poderia combinar os dois e melhorar a robustez nas zonas de incerteza, especialmente para cancelamentos.

4. **Calibração de probabilidades.**  
   Aplicar `CalibratedClassifierCV` para garantir que as probabilidades estimadas são bem calibradas (ex: um voo com 80% de probabilidade de cancelamento deve cancelar ≈ 80% das vezes). Melhora a utilidade dos scores em sistemas de decisão downstream.

### Novas Variáveis

5. **Hora de partida programada (`scheduled_departure_hour`).**  
   Não presente no dataset atual mas disponível na fonte BTS original. Voos matinais têm padrões distintos dos noturnos — seria uma das features mais informativas a adicionar.

6. **Histórico recente do avião e da tripulação.**  
   Dados de *tail number* e *crew rotation* permitiriam modelar o efeito cascata: um atraso num voo anterior aumenta a probabilidade de atraso no seguinte. Esta dinâmica temporal não é capturada pelo modelo atual.

7. **Taxa de ocupação (load factor).**  
   Voos com alta ocupação têm mais pressão para operar e podem ter padrões de cancelamento distintos de voos com poucas reservas.

### Escalabilidade e Deployment

8. **Interface web com Streamlit — FlightSense (implementado).**  
   A aplicação *web* **FlightSense** foi desenvolvida e está em produção em [previsao-cancelamento.streamlit.app](https://previsao-cancelamento.streamlit.app). Permite ao utilizador introduzir os detalhes de um voo (aeroporto de origem, mês, dia, distância) e obter a probabilidade de cancelamento e atraso em tempo real, com *dashboard* de análise exploratória e calculadora de impacto económico. Trabalho futuro nesta componente passa pela integração de explicabilidade *SHAP* por voo diretamente na interface.

9. **API REST para integração operacional.**  
   Expor os modelos como um endpoint REST (FastAPI) que sistemas de gestão aeroportuária possam consultar automaticamente para cada voo programado.

10. **Monitorização de data drift.**  
    Em produção, implementar deteção de *concept drift* — os padrões de 2024 podem não refletir 2025 ou 2026. Ferramentas como Evidently AI permitem monitorizar a estabilidade do modelo ao longo do tempo e alertar quando é necessário re-treino.

---

## 5. Conclusão

Este projeto demonstrou que é possível construir modelos preditivos com valor operacional real para o setor da aviação, utilizando exclusivamente informação disponível antes do voo — sem aceder a dados privilegiados ou em tempo real.

Os dois modelos desenvolvidos — **HistGradient Boosting para cancelamentos** e **XGBoost para atrasos** — não são perfeitos, e as suas limitações foram documentadas com honestidade. O desequilíbrio extremo das classes, a ausência de dados meteorológicos e a circunscrição a um único ano são os principais condicionantes.

Ainda assim, os resultados obtidos constituem uma base sólida e auditável: os modelos são explicáveis via SHAP, foram otimizados com validação cruzada estratificada e os thresholds foram ajustados para maximizar a utilidade prática (F1) em detrimento de uma accuracy enganosamente alta.

O caminho para um sistema de produção está bem definido: dados meteorológicos, hora de partida e histórico do avião são as três melhorias com maior potencial de impacto. Com esses ingredientes, os modelos aqui desenvolvidos constituem um ponto de partida robusto para uma solução escalável de apoio à decisão no setor da aviação.

---

*Coimbra Business School | ISCAC — Licenciatura em Ciência de Dados para a Gestão*  
*Unidade Curricular: Projeto em Ciência de Dados — Docente: Dora Melo*

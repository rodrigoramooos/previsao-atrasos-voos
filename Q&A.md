# Q&A Assíncrono — FlightSense

**Projeto:** Previsão de Atrasos e Cancelamentos de Voos Comerciais  
**Grupo:** 1 — Rodrigo Ramos (a2023137922) · Bruno Almeida (a2023143583)  
**Unidade Curricular:** Projeto em Ciência de Dados para a Gestão · ISCAC · Maio de 2026

---

## Pergunta 1 — Porque é que o ROC-AUC do modelo de cancelamentos (0.871) é mais alto do que o de atrasos (0.723), se o modelo de cancelamentos parece ter pior desempenho na prática?

O ROC-AUC de cancelamentos é aparentemente mais elevado porque a curva ROC é fortemente influenciada pela abundância de negativos verdadeiros — no conjunto de teste existem cerca de 197 000 voos não cancelados que o modelo classifica corretamente de forma trivial. Com um desequilíbrio de 64:1, basta acertar na grande maioria dos negativos para inflar o AUC.

A métrica honesta neste contexto é a **Average Precision (PR-AUC)**, que avalia o modelo apenas nas zonas onde ele tem de tomar decisões difíceis — e aí a vantagem inverte-se claramente: `is_delayed` (AP=0.226) supera `cancelled` (AP=0.100). A curva Precision-Recall não é afetada pelos negativos fáceis e reflete genuinamente a capacidade de deteção em classes desequilibradas. Por esta razão, ao longo do projeto a PR-AUC foi usada como métrica principal de referência na seleção e comparação de modelos.

---

## Pergunta 2 — Porque é que não foram usados dados meteorológicos, sendo que o tempo é a principal causa de cancelamentos?

A decisão foi deliberada e reflete uma restrição operacional real: o objetivo do projeto é prever perturbações **antes da partida**, com informação disponível no momento da previsão. Dados meteorológicos em tempo real não estão disponíveis quando um passageiro ou operador quer saber horas ou dias antes se o voo vai cancelar.

Adicionalmente, o dataset da Bureau of Transportation Statistics (BTS) utilizado não inclui variáveis meteorológicas — integrar fontes externas como NOAA ou OpenWeatherMap implicaria um cruzamento por aeroporto e data que estava fora do âmbito desta fase do projeto.

A análise de falsos negativos confirma esta limitação: os eventos não detetados têm probabilidades medianas de 0.659 (cancelamentos) e 0.455 (atrasos), muito abaixo dos thresholds. O modelo não hesita nestes casos — simplesmente não tem informação suficiente para os identificar. A integração de dados meteorológicos históricos é a melhoria com maior impacto esperado identificada no roadmap do projeto.

---

## Pergunta 3 — Como funciona o TunedThresholdClassifierCV e o que justifica os thresholds de 0.806 e 0.606?

O `TunedThresholdClassifierCV` é um wrapper da scikit-learn que envolve um classificador já treinado e otimiza automaticamente o threshold de decisão em vez de usar o padrão de 0.5. O seu funcionamento divide-se em duas fases:

Na primeira fase, o wrapper percorre sistematicamente um conjunto de thresholds candidatos no intervalo [0, 1]. Para cada candidato, aplica validação cruzada estratificada sobre os dados de treino — a estratificação é essencial para garantir que cada fold mantém a proporção real das classes minoritárias, evitando folds onde os cancelamentos estão sub-representados. Em cada fold, calcula o F1-score resultante de aplicar aquele threshold às probabilidades estimadas. No final, seleciona o threshold que maximiza o F1-score médio entre todos os folds. Todo este processo ocorre exclusivamente sobre os dados de treino, sem qualquer acesso ao conjunto de teste, o que evita data leakage.

Na segunda fase, o threshold selecionado é fixado no modelo final e aplicado ao conjunto de teste para produzir as previsões definitivas.

**Porque 0.806 para cancelamentos?** O desequilíbrio de 64:1 faz com que qualquer threshold baixo gere um volume insustentável de falsos alarmes — por cada cancelamento detetado, o modelo emitiria dezenas de alertas desnecessários. O threshold elevado de 0.806 significa que o modelo só sinaliza um voo quando a probabilidade estimada é muito alta, controlando o rácio FP/TP em 8.0. A análise de threshold confirmou que o pico de F1 é estreito neste valor: baixar o threshold faz crescer os falsos positivos muito mais depressa do que os verdadeiros positivos.

**Porque 0.606 para atrasos?** O menor desequilíbrio (11:1) permite um threshold mais baixo sem colapso da Precision. O pico de F1 é mais largo e existe margem real de ajuste — o modelo consegue detetar 4 em cada 10 atrasos reais com um rácio FP/TP de 3.5, substancialmente mais eficiente do que o modelo de cancelamentos. Um operador pode ainda calibrar este valor conforme o seu contexto: subir o threshold para maior Precision (menos alertas, mais fiáveis) ou baixá-lo para maior Recall (mais cobertura, mais falsos alarmes).

---

*Coimbra Business School | ISCAC — Licenciatura em Ciência de Dados para a Gestão · Maio de 2026*

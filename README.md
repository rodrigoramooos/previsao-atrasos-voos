# Análise e Previsão de Cancelamentos e Atrasos em Voos Comerciais

Licenciatura em Ciência de Dados para a Gestão — Coimbra Business School | ISCAC  
Unidade Curricular: Projeto em Ciência de Dados — Docente: Dora Melo (dmelo@iscac.pt)  
Ano Letivo: 2025/2026

---

## Identificação da Equipa

- **Grupo nº:** 1
- **Membros:**
  - Rodrigo Ramos — a2023137922
  - Bruno Almeida — a2023143583

---

## Organização do Repositório

A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:

```
previsao-atrasos-voos/
├── app/                    ← Aplicação Streamlit (FlightSense)
│   ├── .streamlit/
│   │   └── config.toml
│   ├── app.py
│   ├── feature_names.json
│   ├── modelo_cancelamentos_voos.pkl
│   ├── modelo_atrasos_voos.pkl
│   └── requirements.txt
├── data/
│   ├── raw/
│   │   └── dataset-original.md               ← Descrição e link do dataset (Kaggle)
│   └── processed/
│       └── flight_data_processed_github.csv  ← Dataset processado (pós-EDA)
├── docs/                   ← Documentação técnica detalhada por Milestone
│   ├── M1_iniciacao.md
│   ├── M2_exploracao.md
│   ├── M3_modelacao.md
│   └── M4_conclusoes.md
├── notebooks/              ← Versões exportadas do Kaggle Code
│   ├── 1.0_eda_limpeza.ipynb          ← Corresponde à Fase 2
│   ├── 2.0_modelacao_treino.ipynb     ← Corresponde à Fase 3
│   └── 3.0_interpretacao.ipynb        ← Corresponde à Fase 4
├── reports/
│   └── figures/            ← Figuras geradas nos notebooks (36 ficheiros)
├── src/                    ← Reservado para módulos auxiliares
├── index.html              ← Página de entrada (GitHub Pages)
├── requirements.txt        ← Dependências do projeto
└── README.md
```

---

## 1. Iniciação (Milestone 1)

### Contexto e Problema de Negócio

O transporte aéreo comercial é um setor de elevada complexidade operacional onde cancelamentos e atrasos de voos geram impactos financeiros significativos para companhias aéreas, aeroportos e passageiros. Nos Estados Unidos, dados do *Bureau of Transportation Statistics* indicam que perturbações operacionais custam anualmente milhares de milhões de dólares à indústria, não só em compensações diretas, mas também em custos de realojamento, afetação de tripulações e deterioração da experiência do passageiro.

O problema identificado consiste na ausência de instrumentos preditivos capazes de antecipar, **antes da partida**, se um determinado voo será cancelado ou sofrerá um atraso significativo. A gestão reativa destas perturbações — agir apenas após a ocorrência — é operacionalmente ineficaz e economicamente prejudicial.

### Relevância do Projeto

A antecipação de cancelamentos e atrasos tem impacto direto em:

- **Eficiência operacional** das companhias aéreas — realocação atempada de tripulações e aeronaves
- **Experiência e satisfação dos passageiros** — notificações preventivas reduzem surpresas
- **Custos logísticos e de gestão aeroportuária** — planeamento de *gates* e pessoal em dias de maior risco
- **Planeamento de rotas e alocação de recursos** — identificação de aeroportos e períodos de maior vulnerabilidade

### Objetivos SMART do Projeto

Desenvolver e avaliar dois modelos de classificação binária — um para prever o cancelamento de voo e outro para prever atraso significativo igual ou superior a 15 minutos — utilizando dados históricos de 1 041 151 voos comerciais norte-americanos de 2024, com *ROC-AUC* superior a 0.80 no modelo de cancelamentos e superior a 0.70 no modelo de atrasos, avaliados também por F1-*score* e *Avg Precision* em validação cruzada estratificada com cinco partições, até ao final da unidade curricular de Projeto em Ciência de Dados (maio de 2026).

### Fonte de Dados

- **Dataset:** [Flight Delay & Cancellation Data — 1M+ (2024)](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data) — *Bureau of Transportation Statistics* (BTS), U.S. DOT
- **Notebook EDA:** [1.0_eda_limpeza — Kaggle](https://www.kaggle.com/code/rodrigoramooos/eda-previsao-de-cancelamentos-em-voos)
- **Notebook Modelação:** [mod-final — Kaggle](https://www.kaggle.com/code/rodrigoramooos/modelacao-previsao-de-cancelamentos-em-voos)
- **Dimensão do Dataset:** 1 048 575 linhas · 18 colunas

### Perguntas de Investigação

1. Quais as variáveis com maior importância na previsão de atrasos e cancelamentos de voos, de acordo com os modelos de *machine learning* utilizados?
2. É possível prever o cancelamento de um voo antes da sua partida com desempenho satisfatório (ex.: F1-*score* superior a 0.70)?
3. Variáveis temporais (hora do dia, dia da semana, mês) apresentam impacto significativo na probabilidade de cancelamento ou atraso de voos?
4. Existem diferenças significativas na taxa de cancelamento entre aeroportos de origem e estados, e estas diferenças são captadas pelo modelo?
5. A inclusão de variáveis operacionais (ex.: `taxi_out`, `air_time`, `delays`) melhora o desempenho preditivo dos modelos quando comparado com modelos base?

### Ferramentas e Bibliotecas Python

- **pandas** — Carregamento, limpeza e transformação de dados
- **numpy** — Operações numéricas e manipulação de *arrays*
- **matplotlib / seaborn / plotly** — Visualização de dados estatísticos e interativos
- **scikit-learn** — Pré-processamento, modelação, validação cruzada e `TunedThresholdClassifierCV`
- **XGBoost** — Algoritmo de *gradient boosting* para o modelo de atrasos
- **SHAP** — Interpretabilidade global e local dos modelos
- **Streamlit** — Desenvolvimento da aplicação web *FlightSense*
- **joblib** — Serialização e carregamento dos modelos treinados

📄 Documento completo: [`docs/M1_iniciacao.md`](docs/M1_iniciacao.md)

---

## 2. Exploração (Milestone 2)

### Limpeza e Preparação

- Foram analisados os valores em falta, verificando-se que estão maioritariamente associados a voos cancelados — mantidos como informação estruturalmente coerente com o fenómeno.
- Foram removidos registos duplicados (~7 400) e valores fisicamente impossíveis (~23 registos com velocidades de cruzeiro impossíveis).
- Variáveis com risco de *data leakage* (`weather_delay`, `taxi_out`, `late_aircraft_delay`, `dep_time`, `air_time`, etc.) foram excluídas do conjunto de modelação por não estarem disponíveis antes da partida.
- Foi realizada engenharia de atributos com criação de novas variáveis — `is_long_flight` (> 1 500 milhas), `is_short_flight` (< 300 milhas) e `is_weekend` — para melhorar a capacidade explicativa dos modelos.

### Desequilíbrio das Classes

| Variável-alvo | Classe positiva | Frequência |
|---|---|---|
| `cancelled` | Voo cancelado | **1.53%** |
| `is_delayed` | Atraso ≥ 15 min | **8.52%** |

A variável `is_delayed` foi construída a partir de `weather_delay + late_aircraft_delay ≥ 15 min`, excluindo voos cancelados.

### Principais Conclusões (EDA)

- `day_of_month` é a variável com maior variabilidade em ambas as variáveis-alvo — dias específicos concentram sistematicamente mais perturbações.
- `month` revela sazonalidade clara: janeiro e fevereiro, correspondentes ao inverno norte-americano, elevam o risco de cancelamento.
- Aeroportos de origem com influência significativa: Buffalo (BUF) e Chicago Midway (MDW) associados a maior risco; grandes *hubs* como Atlanta (ATL) funcionam como fator protetor.
- Variáveis como a distância, de forma isolada, não distinguem claramente voos perturbados dos restantes — a combinação de atributos temporais e geográficos é essencial.
- O forte desequilíbrio da classe `cancelled` (1.53%) exige métricas adequadas como *Recall*, *Precision*, F1-*score* e *Avg Precision* (PR-AUC) na fase de modelação.

📄 Documento completo: [`docs/M2_exploracao.md`](docs/M2_exploracao.md)

---

## 3. Modelação (Milestone 3)

### Estratégia

- **Métrica de seleção principal:** *Avg Precision* (PR-AUC) — mais adequada que *ROC-AUC* em classes muito desequilibradas
- **Otimização de *threshold*:** `TunedThresholdClassifierCV` com maximização de F1
- **Validação:** `StratifiedKFold` (k=5) para respeitar o desequilíbrio
- **Gestão de desequilíbrio:** `class_weight='balanced'` (scikit-learn) e `scale_pos_weight` (XGBoost)
- **Modelos candidatos:** Regressão Logística (linha de base), *HistGradient Boosting*, XGBoost

### Síntese Executiva dos Modelos

| | Cancelamentos | Atrasos |
|---|---|---|
| **Algoritmo** | *HistGradient Boosting* | XGBoost |
| **F1-*score*** | 0.151 | 0.289 |
| ***Recall*** | 0.266 | 0.396 |
| ***Precision*** | 0.105 | 0.227 |
| ***ROC-AUC*** | **0.854** ✓ | **0.718** ✓ |
| ***Avg Precision* (PR-AUC)** | 0.087 | 0.225 |
| ***Threshold* otimizado** | 0.806 | 0.613 |
| **Ficheiro** | `modelo_cancelamentos_voos.pkl` | `modelo_atrasos_voos.pkl` |

Ambos os objetivos *ROC-AUC* foram superados (> 0.80 nos cancelamentos; > 0.70 nos atrasos).

### Interpretação dos Modelos (SHAP)

**Cancelamentos — principais fatores:**

| Variável | Interpretação |
|---|---|
| `month` | Meses de inverno (janeiro/fevereiro) aumentam substancialmente o risco |
| `day_of_month` | O dia específico tem impacto extremo — volatilidade muito elevada |
| `distance` | Voos mais longos apresentam maior probabilidade de cancelamento |
| Origens (ATL, DFW, CLT) | Grandes *hubs* reduzem o risco; aeroportos regionais como BUF aumentam-no |

**Atrasos — principais fatores:**

| Variável | Interpretação |
|---|---|
| `day_of_month` | Fator mais influente — dias específicos determinam a pontualidade do voo |
| `month` | Padrão idêntico ao dos cancelamentos: inverno aumenta o risco |
| `distance` | Voos curtos têm mais risco de atraso; voos longos, menos |
| `origin_DFW` | Dallas-Fort Worth aumenta drasticamente o risco de atraso |

### Impacto Prático

O modelo de cancelamentos identifica corretamente **1 em cada 4 voos que seriam cancelados** antes de qualquer perturbação acontecer. O modelo de atrasos sinaliza cerca de **4 em cada 10 voos que chegarão com atraso**. A principal utilidade está na redução de falsos negativos — cada cancelamento antecipado tem valor operacional concreto; um alerta preventivo adicional (falso positivo) tem custo muito inferior a um cancelamento não detetado.

📄 Documento completo: [`docs/M3_modelacao.md`](docs/M3_modelacao.md)

---

## 4. Finalização (Milestone 4)

### Resposta ao Problema

Os dois modelos desenvolvidos respondem à questão central do projeto — é possível antecipar perturbações operacionais antes da partida de um voo, utilizando exclusivamente informação disponível em pré-voo, sem acesso a dados meteorológicos em tempo real. O *ROC-AUC* de 0.854 no modelo de cancelamentos supera o objetivo definido (> 0.80); o *ROC-AUC* de 0.718 no modelo de atrasos supera igualmente o objetivo (> 0.70). Os modelos são explicáveis via SHAP e estão disponíveis em produção na aplicação [FlightSense](https://previsao-cancelamento.streamlit.app).

### Recomendações de Inovação

1. **Integrar dados meteorológicos históricos (NOAA / OpenWeatherMap)** — maior ganho de *Avg Precision* esperado, abordando a principal limitação atual
2. **Adicionar hora de partida programada** (`scheduled_departure_hour`) — variável altamente informativa ausente do dataset
3. **Testar *ensemble stacking*** (HistGBT + XGBoost) — melhor robustez nas zonas de incerteza
4. **Aplicar calibração de probabilidades** (`CalibratedClassifierCV`) — *scores* mais úteis em sistemas de decisão automatizados
5. **Expor os modelos como API REST** (FastAPI) para integração com sistemas de gestão aeroportuária

📄 Documento completo: [`docs/M4_conclusoes.md`](docs/M4_conclusoes.md)

---

## Fonte de Dados

- **Dataset:** [Flight Delay and Cancellation Data (Kaggle)](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data)
- **Origem:** *Bureau of Transportation Statistics* (BTS) — U.S. Department of Transportation
- **Dimensão:** 1 048 575 linhas · 18 colunas

---

## Referências

- **Nadeem, A. (2024).** *Flight Delay & Cancellation Data (1 Million+ 2024)*. Kaggle. [Dataset](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024)
- **Bureau of Transportation Statistics — BTS. (2024).** *Marketing Carrier On-Time Performance Data*. U.S. Department of Transportation. Disponível em: [transtats.bts.gov](https://www.transtats.bts.gov)
- **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30. [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

---

## Como Reproduzir este Projeto

```bash
# 1. Clonar o repositório
git clone https://github.com/rodrigoramooos/previsao-atrasos-voos.git
cd previsao-atrasos-voos

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar os notebooks por ordem
jupyter notebook notebooks/1.0_eda_limpeza.ipynb        # Fase 2 — EDA e pré-processamento
jupyter notebook notebooks/2.0_modelacao_treino.ipynb   # Fase 3 — Modelação e avaliação
jupyter notebook notebooks/3.0_interpretacao.ipynb      # Fase 4 — Interpretabilidade SHAP
```

> **Nota:** O dataset original (≈ 1 M linhas) está alojado no Kaggle. O ficheiro `data/processed/flight_data_processed_github.csv` contém uma amostra pós-processamento compatível com os notebooks. Para replicar na íntegra, descarregar o dataset original via [Kaggle](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024) e colocar em `data/raw/`.

---

**Instituição:** [Coimbra Business School | ISCAC](https://www.iscac.pt)  
**Curso:** Licenciatura em Ciência de Dados para a Gestão  
**Unidade Curricular:** Projeto em Ciência de Dados  
**Docente Responsável:** Dora Melo (dmelo@iscac.pt)  
**Ano Letivo:** 2025/2026


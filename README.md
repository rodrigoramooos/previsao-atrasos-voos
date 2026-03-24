# Análise e Previsão de Atrasos e Cancelamentos de Voos Comerciais

## Identificação da Equipa
- **Grupo nº:** 1  
- **Membros:**
  - Rodrigo Ramos — a2023137922  
  - Bruno Almeida — a2023143583  

## Organização do Repositório
A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:
* **`data/`**: Armazenamento de dados (dados brutos em `raw/` e processados em `processed/`).
* **`docs/`**: Documentação técnica detalhada dividida por Milestones (M1, M2 e M3).
* **`notebooks/`**: Jupyter Notebooks para experimentação, limpeza e modelação.
* **`src/`**: Código-fonte modular (scripts `.py`) para funções reutilizáveis.
* **`reports/`**: Relatórios finais, apresentações e exportação de figuras (`figures/`).
* **`requirements.txt`**: Ficheiro de configuração com as bibliotecas necessárias.
* **`README.md`**: Guia principal do projeto.

## 1. Iniciação (Milestone 1)
### Contexto e Problema de Negócio
Este projeto tem como objetivo prever atrasos e cancelamentos em voos comerciais, recorrendo a técnicas de ciência de dados e aprendizagem automática aplicadas a dados reais de operações aéreas.
O problema identificado prende-se com a dificuldade em antecipar disrupções operacionais no setor da aviação, que impactam negativamente a eficiência das companhias aéreas, a gestão aeroportuária e a experiência dos passageiros.
Através da análise de dados reais de voos realizados em 2024, pretende-se desenvolver um modelo preditivo capaz de identificar padrões associados a atrasos e cancelamentos, permitindo apoiar a tomada de decisão, melhorar o planeamento operacional e reduzir custos logísticos.
### Relevância do Projeto
Os atrasos em voos têm impacto direto em:
* Eficiência operacional das companhias aéreas
* Experiência e satisfação dos passageiros
* Custos logísticos e de gestão aeroportuária
* Planeamento de rotas e alocação de recursos
A previsão antecipada de atrasos permite apoiar decisões operacionais, reduzir custos e melhorar a qualidade do serviço.
### Objetivos SMART do Projeto
- **S (Specific):** Prever se um voo será cancelado ou sofrerá atraso significativo (≥15 minutos), com base em dados históricos de operações aéreas.
- **M (Measurable):** Avaliar o desempenho dos modelos através de métricas como Accuracy, Precision, Recall e F1-score, comparando diferentes algoritmos de machine learning.
- **A (Achievable):** Utilização de um dataset real com mais de 1 milhão de registos, adequado à aplicação de técnicas de análise de dados e modelação preditiva.
- **R (Relevant):** Contribuir para a melhoria da eficiência operacional no setor da aviação, apoiando a tomada de decisão e a redução de custos associados a atrasos e cancelamentos.
- **T (Time-bound):** Desenvolver, treinar e avaliar os modelos preditivos até ao final da unidade curricular.
### Fonte de Dados
* **Dataset:** (https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data)
* **Kaggle Code:** (https://www.kaggle.com/code/rodrigoramooos/projeto-previs-o-de-atraso-e-cancelamento-de-voos)
* **Dimensão do Dataset:** [1.048.575 linhas, 18 colunas]
### Perguntas de Investigação
1. Quais as variáveis com maior importância na previsão de atrasos e cancelamentos de voos, de acordo com os modelos de machine learning utilizados?
2. É possível prever o cancelamento de um voo antes da sua partida com desempenho satisfatório (ex.: F1-score superior a 0.70)?
3. Variáveis temporais (hora do dia, dia da semana, mês) apresentam impacto significativo na probabilidade de cancelamento ou atraso de voos?
4. Existem diferenças significativas na taxa de cancelamento entre aeroportos de origem e estados, e estas diferenças são captadas pelo modelo?
5. A inclusão de variáveis operacionais (ex.: taxi_out, air_time, delays) melhora o desempenho preditivo dos modelos quando comparado com modelos base?
### Ferramentas e bibliotecas Python
* **pandas** — Carregamento, limpeza e transformação de dados.
* **numpy** — Operações numéricas e manipulação de arrays. 
* **matplotlib** — Criação de gráfico base e customização
* **seaborn** — Visualização de dados estatísticos complexos
* **scikit-learn** — Pré-processamento de dados e desenvolvimento de modelos de Machine Learning.


## 2. Exploração (Milestone 2)

### Limpeza e Preparação
* Foram analisados e interpretados os valores em falta, verificando-se que estes estão maioritariamente associados a voos cancelados, sendo por isso mantidos como informação relevante.  
* Foram removidos registos duplicados e valores fisicamente impossíveis, garantindo a integridade dos dados.  
* Variáveis com risco de *data leakage* (ex: `weather_delay`, `taxi_out`, `late_aircraft_delay`) foram excluídas do conjunto de modelação por não estarem disponíveis no momento da previsão.  
* Foi realizada seleção de atributos e criação de novas variáveis, como `is_long_flight` e `flight_period`, com o objetivo de melhorar a capacidade explicativa do dataset.  
* Detalhes completos disponíveis em `docs/M2_exploracao.md`.

---

### Principais Conclusões (EDA)

<img width="812" height="494" alt="image" src="https://github.com/user-attachments/assets/0d78763a-f8ed-4378-8667-11626d6b371f" />


                                                (Figura 1 - Distribuição da variável alvo `cancelled`)

* A variável alvo apresenta um forte desequilíbrio (≈97,8% voos não cancelados vs ≈2,2% cancelados), o que implica a utilização de métricas adequadas na modelação.  
* Algumas variáveis inicialmente disponíveis não são adequadas para previsão por introduzirem *data leakage*, sendo necessária uma seleção cautelosa de atributos. 
* Variáveis como a distância, de forma isolada, não permitem distinguir claramente voos cancelados e não cancelados, evidenciando a necessidade de engenharia de atributos e combinação de variáveis.  

## 3. Modelação (Milestone 3)
### Abordagem Técnica
* **Modelos:** [Ex: Random Forest e XGBoost]
* **Métrica Principal:** [Ex: F1-Score ou RMSE]

## 4. Finalização (Milestone 4)
### Resposta ao Problema
[Resumo da solução e como ela gera valor para o negócio.]
### Recomendações de Inovação
1. [Sugestão prática baseada nos resultados]

## Fonte de Dados
* **Dataset:** [Flight Delay and Cancellation Data (Kaggle)](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data)
* **Dimensão do Dataset:** 1.048.575 linhas | 18 colunas
* **Fonte do código:** [Link para o Notebook](https://www.kaggle.com/code/rodrigoramooos/projeto-previsao-de-atraso-e-cancelamento-de-voos)

## Referências

### Citação do Dataset e Bases Oficiais
* **Nadeem, A. (2024).** *Flight Delay & Cancellation Data (1 Million+ 2024)*. Kaggle. [Link para o Dataset](https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024).
* **Bureau of Transportation Statistics (BTS). (2024).** *Marketing Carrier On-Time Performance Data*. U.S. Department of Transportation. [Link para a origem dos dados](https://transtats.bts.gov/Marketing_Monthly.aspx5ry_lrn4=FDFH&N44_Qry=E&5ry_Pn44vr4=DDD&5ry_Nv42146=DDD&heY_fryrp6lrn4=FDFI&heY_fryrp6Z106u=EE)



## Como Reproduzir este Projeto
1. Clone o repositório: `git clone [url-do-repo]`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` seguindo a ordem numérica.

**Instituição:** [Coimbra Business School | ISCAC](https://www.iscac.pt)  

**Curso:** Licenciatura em Ciência de Dados para a Gestão  

**Unidade Curricular:** Projeto em Ciência de Dados

**Docente Responsável:** Dora Melo (dmelo@iscac.pt)  

**Ano Letivo:** 2025/2026


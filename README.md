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
* **Dimensão:** [1.048.575 linhas, 18 colunas]
### Perguntas de Investigação
1. Quais as variáveis com maior importância na previsão de atrasos e cancelamentos de voos, de acordo com os modelos de machine learning utilizados?
2. É possível prever o cancelamento de um voo antes da sua partida com desempenho satisfatório (ex.: F1-score superior a 0.70)?
3. Variáveis temporais (hora do dia, dia da semana, mês) apresentam impacto significativo na probabilidade de cancelamento ou atraso de voos?
4. Existem diferenças significativas na taxa de cancelamento entre aeroportos de origem e estados, e estas diferenças são captadas pelo modelo?
5. A inclusão de variáveis operacionais (ex.: taxi_out, air_time, delays) melhora o desempenho preditivo dos modelos quando comparado com modelos base?
### Ferramentas e bibliotecas Python
#### Manipulação e preparação de dados
* **pandas** — Carregamento, limpeza e transformação de dados.
* **numpy** — Operações numéricas e manipulação de arrays. 
#### Análise exploratória de dados (EDA)
* **scipy** — Testes estatísticos e validação de hipóteses  
#### Visualização de dados
* **matplotlib** — Criação de gráfico base e customização
* **seaborn** — Visualização de dados estatísticos complexos
* **plotly** — Dashboards e gráficos interativos  
#### Machine Learning
* **scikit-learn**
  * Pré-processamento e normalização
  * Divisão de dados (Train/Test Split)  
  * Implementação de modelos (Logistic Regression, Random Forest, etc.)
#### Avaliação de modelos
* **sklearn.metrics**
  * accuracy  
  * precision  
  * recall  
  * F1-score  
  * ROC-AUC 

## 2. Exploração (Milestone 2)
### Limpeza e Preparação
* [Breve resumo das ações de limpeza tomadas. Detalhes em `docs/M2_exploracao.md`]
### Principais Conclusões (EDA)
> *Dica: Insere aqui o gráfico mais importante do projeto.*
* **Ponto-chave:** [Ex: Identificámos que o fator X influencia em 40% o resultado Y, por aplicação
do método ganho de informação]

## 3. Modelação (Milestone 3)
### Abordagem Técnica
* **Modelos:** [Ex: Random Forest e XGBoost]
* **Métrica Principal:** [Ex: F1-Score ou RMSE]

## 4. Finalização (Milestone 4)
### Resposta ao Problema
[Resumo da solução e como ela gera valor para o negócio.]
### Recomendações de Inovação
1. [Sugestão prática baseada nos resultados]
## Como Reproduzir este Projeto
1. Clone o repositório: `git clone [url-do-repo]`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` seguindo a ordem numérica.

**Instituição:** Coimbra Business School | ISCAC
**Curso:** Licenciatura em Ciência de Dados para a Gestão
**Unidade Curricular:** Projeto em Ciência de Dados
**Professor Responsável:** Dora Melo (dmelo@iscac.pt) 

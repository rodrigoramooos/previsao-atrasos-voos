# [Previsão de Atrasos em Voos (Logística e Transporte]

## Identificação da Equipa
* **Grupo nº:** [1]
* **Membros:**
 * Rodrigo Ramos - a2023137922
 * Bruno Almeida - a2023143583

## Organização do Repositório
A estrutura deste projeto segue as boas práticas de Ciência de Dados e Engenharia de Software:
* **`data/`**: Armazenamento de dados (dados brutos em `raw/` e processados em `processed/`).
* **`docs/`**: Documentação técnica detalhada dividida por Milestones (M1, M2 e M3).
* **`notebooks/`**: Jupyter Notebooks para experimentação, limpeza e modelação.
* **`src/`**: Código-fonte modular (scripts `.py`) para funções reutilizáveis.
* **`reports/`**: Relatórios finais, apresentações e exportação de figuras (`figures/`).
* **`requirements.txt`**: Ficheiro de configuração com as bibliotecas necessárias.

## 1. Iniciação (Milestone 1)
### Contexto e Problema de Negócio
Este projeto baseia-se na previsão de atrasos e cancelamento de voos comerciais, recorrendo a técnicas de ciência de dados e aprendizagem automática aplicadas a dados reais de operações aéreas.
### Relevância do Projeto
Os atrasos em voos têm impacto direto em:
* Eficiência operacional das companhias aéreas
* Experiência e satisfação dos passageiros
* Custos logísticos e de gestão aeroportuária
* Planeamento de rotas e alocação de recursos
A previsão antecipada de atrasos permite apoiar decisões operacionais, reduzir custos e melhorar a qualidade do serviço.
### Objetivos SMART do Projeto
* **S - Specific:** [Prever se um voo irá sofrer atraso significativo (ex.: ≥15 minutos).]
* **M - Measurable:** [Avaliar o desempenho do modelo através de Accuracy, F1-score, ...]
* **A - Achievable:** [Utilização de dataset real com 1M+ de registos.]
* **R - Relevant:** [Aplicação direta em logística aérea e otimização operacional.]
* **T - Time-bound:** [Desenvolvimento do modelo e avaliação até ao final da unidade curricular.]
### Fonte de Dados
* **Dataset:** (https://www.kaggle.com/datasets/nalisha/flight-delay-and-cancellation-data-1-million-2024/data)
* **Dimensão:** [1.048.575 linhas, 18 colunas]
### Perguntas de Investigação
* 1. Quais os principais fatores que contribuem para o atraso de voos?
* 2. É possível prever com precisão se um voo irá atrasar antes da sua partida?
* 3. Existem padrões temporais (hora do dia, dia da semana, estação) associados a maiores probabilidades de atraso?
* 4. Certas companhias aéreas ou aeroportos apresentam maior probabilidade de atrasos?
* 5. Que variáveis têm maior influência na ocorrência de atrasos e de que forma contribuem para a sua previsão?
### Ferramentas e bibliotecas Python
#### Manipulação e preparação de dados
* **pandas** — carregamento, limpeza e transformação de dados.
* **numpy** — operações numéricas e manipulação de arrays. 
#### Análise exploratória de dados (EDA)
* **scipy** — análise estatística  
#### Visualização de dados
* **matplotlib** — gráficos base  
* **seaborn** — visualizações estatísticas  
* **plotly** — gráficos interativos  
#### Machine Learning
* **scikit-learn**
  * preparação de dados  
  * divisão treino/teste  
  * modelos base (Logistic Regression, Random Forest, etc.)
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

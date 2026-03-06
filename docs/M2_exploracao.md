# Milestone 2: Análise Exploratória e Engenharia de Atributos
> **Nota de Revisão:** Este documento pressupõe que o dataset já foi identificado e descrito no
ficheiro `docs/M1_iniciacao.md`. Caso precise de consultar o significado original das variáveis,
deve consultar essa Milestone.
## 1. Análise Exploratória de Dados (EDA)
### 1.1. Distribuição da Variável Alvo
*A nossa variável alvo é a cancelled, que indica se um voo foi ou não cancelado. Esta variável apresenta muito desequilíbrio , o que é normal em problemas de deteção de falhas ou eventos raros na aviação. Por ser uma variável binária, segue uma distribuição de Bernoulli altamente assimétrica.*
> **Factos importantes:** A variável alvo está desequilibrada, com 1025269 voos operados ou seja 97,78% (Classe 0) e apenas 23306 de voos cancelados, o que se traduz em 2,22% (Classe 1).
### 1.2. Correlações Relevantes
* **Atributo Origin (Aeroporto) vs. Alvo:** Notámos que a operação está muito concentrada em grandes aeroportos (como ATL e DFW), implica que a infraestrutura e o volume de tráfego do aeroporto de origem têm maior influência na probabilidade de disrupção do que a rota em si, assim influenciando os voos cancelados.
(Referência: Gráfico de Barras "Top 10 Aeroportos de Origem")
* **Atributo Weather Delay vs. Alvo:** Foi observado que a grande maioria dos voos apresenta weather_delay = 0 e que os cancelamentos se concentram exclusivamente neste grupo. Isto quer dizer que o atraso meteorológico registado é uma informação posterior ao evento (só existe se o voo operou), não sendo utilizada como causa direta detetável para prever o cancelamento no momento da partida.
(Referência: Countplot "Cancelamento vs Existência de Weather Delay")
* **Atributo Distance vs. Alvo:** Através da visualização por Violin Plot, verificamos que a distribuição da distância percorrida é idêntica para voos realizados e cancelados. Isto indica que a distância (distância da rota agendada) não possui uma relação linear ou determinante com a probabilidade de um voo ser cancelado, sendo este evento mais dependente de fatores externos ou operacionais do que da extensão da rota.
(Referência: Violin Plot "Distribuição da Distância por Cancelamento")
## 2. Qualidade dos Dados e Limpeza
### 2.1. Tratamento de Dados em Falta (Missing Data)
* **Colunas afetadas:** [Lista de colunas]
* **Estratégia adotada:** (Ex: "Substituímos os nulos da coluna 'Salário' pela mediana para
evitar o impacto de outliers.")
### 2.2. Outliers e Inconsistências
*Descrevam se encontraram valores impossíveis (ex: idade = 200) e como os resolveram.*
## 3. Engenharia de Atributos (Feature Engineering)
### 3.1. Transformações Realizadas
* **Encoding:** (Ex: "Convertemos a variável 'Género' em numérica usando One-Hot Encoding.")
* **Escalonamento:** (Ex: "Aplicámos o StandardScaler nas variáveis numéricas para que todas
fiquem na mesma escala.")
### 3.2. Criação de Novos Atributos
*Descrevam as variáveis que criaram para ajudar o modelo.*
* **Nova Variável [Nome]:** (Ex: "Criámos a 'Tenure_Per_Year' que divide o tempo de contrato
pela idade do cliente.")
## 4. Dicionário de Dados Final (Pós-Processamento)
*Listagem final das variáveis que serão entregues ao modelo na Fase 3.*
| Atributo | Tipo | Descrição |
| :--- | :--- | :--- |
| `cliente_id` | ID | Removido (não preditivo) |
| `idade_norm` | Float | Idade após normalização |
| `is_premium` | Binary | 1 para clientes com plano superior |
## 5. Conclusões da Fase de Exploração
*O que aprenderam sobre o dataset que não sabiam no final do Milestone 1? Os dados são suficientes
para avançar para a modelação?*
---
*Data de última atualização: [DD/MM/AAAA]* 

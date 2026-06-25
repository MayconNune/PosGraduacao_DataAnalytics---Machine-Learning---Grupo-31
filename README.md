# Wine Quality Prediction with Machine Learning

Classificação binária da qualidade de vinhos tintos portugueses (Vinho Verde) a partir de características físico-químicas. Projeto desenvolvido para o Tech Challenge — Fase 2 da Pós-Graduação em Data Analytics (POSTECH), Grupo 31.

---

## 1. Problema

O dataset original contém notas de qualidade de 3 a 8, atribuídas por especialistas. Para este projeto, a variável alvo foi binarizada:

- **Alta Qualidade (1)**: nota original ≥ 7
- **Qualidade Comum (0)**: nota original < 7

**Desafio**: classes desbalanceadas — apenas ~19% das amostras são de alta qualidade.

---

## 2. Dataset

**Fonte**: [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

- 1143 amostras de vinho tinto portugês (Vinho Verde)
- 11 variáveis preditoras físico-químicas + qualidade (target)
- **Features**: acidez fixa, acidez volátil, ácido cítrico, açúcar residual, cloretos, dióxido de enxofre livre, dióxido de enxofre total, densidade, pH, sulfatos, teor alcoólico

---

## 3. Análise Exploratória (EDA)

A EDA completa está disponível em `notebooks/Analise_Base_Vinhos.ipynb`. Foram gerados 6+ gráficos para suportar as decisões analíticas descritas abaixo.

### 3.1 Carregamento e inspeção inicial

O dataset `WineQT.csv` foi carregado com 1143 linhas e 13 colunas (12 features + `Id`). Todas as colunas apresentaram tipos de dados corretos e **nenhum valor nulo** foi encontrado. A coluna `quality` possui apenas 6 valores únicos (3 a 8), indicando que as notas são discretas e concentradas.

### 3.2 Limpeza dos dados

- Remoção da coluna `Id` (identificador sequencial sem valor preditivo)
- Identificação de **125 linhas duplicadas** após a remoção do `Id` — registros com valores físico-químicos idênticos mas com IDs diferentes
- Decisão: **remoção das duplicatas**, pois não agregam informação nova ao modelo e podem causar vazamento de dados entre treino e teste
- Base final: **1018 registros**
- Dados exportados para `data/base_wine_limpa.csv`

### 3.3 Distribuição da qualidade (Gráficos 1 e 2)

**Gráfico 1** — gráfico de barras da variável `quality`:
**Gráfico 2** — histograma da variável `quality`.

A distribuição é concentrada nas notas 5 e 6 (~77% das amostras). Não há vinhos com nota abaixo de 3 nem acima de 8, indicando critérios de avaliação rigorosos.

| Qualidade | Contagem | % |
|---|---|---|
| 3 | 7 | 0,7% |
| 4 | 36 | 3,5% |
| 5 | 382 | 37,6% |
| 6 | 398 | 39,1% |
| 7 | 155 | 15,2% |
| 8 | 40 | 3,9% |

Foi criada a coluna binária `cluster_quality`:
- **Alta Qualidade (1)**: nota ≥ 7 → **159 vinhos (~19%)**
- **Qualidade Comum (0)**: nota < 7 → **859 vinhos (~81%)**

**Decisão**: manter o desbalanceamento natural para tratamento via SMOTE nos modelos, em vez de balanceamento artificial na EDA.

### 3.4 Estatísticas descritivas

O resumo estatístico (`df.describe()`) revelou:
- Desvio padrão elevado em `residual sugar` (1,36 g/L), sugerindo heterogeneidade entre os vinhos
- Média de `alcohol` em 10,44% com range de 8,4 a 14,9
- Média de `volatile acidity` em 0,53 g/L com máximo de 1,58 g/L

### 3.5 Investigação de hipóteses

**Hipótese: açúcar residual influencia a qualidade?**

Foi gerado um **boxplot de `residual sugar` por `quality`** e calculada a correlação (r = 0,02). **Conclusão**: açúcar residual isoladamente **não é um fator preditivo** de qualidade.

### 3.6 Matriz de correlação (Gráfico 3)

**Gráfico 3** — heatmap da matriz de correlação entre todas as variáveis físico-químicas.

**Correlações com a qualidade (target):**

| Variável | Correlação (r) | Interpretação |
|---|---|---|
| Teor alcoólico | **+0,49** | Quanto maior o álcool, maior a qualidade |
| Acidez volátil | **-0,41** | Quanto menor a acidez volátil, maior a qualidade |
| Sulfatos | +0,26 | Correlação positiva moderada |
| Ácido cítrico | +0,24 | Correlação positiva moderada |
| Densidade | -0,17 | Correlação negativa fraca |
| Cloretos | -0,13 | Correlação negativa fraca |
| pH | -0,06 | Correlação muito fraca |
| Dióxido de enxofre livre | -0,05 | Correlação muito fraca |
| Açúcar residual | -0,03 | Correlação praticamente nula |
| Acidez fixa | +0,01 | Correlação praticamente nula |
| Dióxido de enxofre total | -0,01 | Correlação praticamente nula |

**Gráfico 4** — barras da força absoluta de correlação (`|r|`), ordenando todas as variáveis por impacto na qualidade.

### 3.7 Análise das variáveis mais correlacionadas

**Top 5 positivas** (Gráfico 5 — scatter `alcohol` vs `quality`):
1. Álcool (+0,49)
2. Sulfatos (+0,26)
3. Ácido cítrico (+0,24)
4. Acidez fixa (+0,12)
5. Açúcar residual (+0,02)

**Top 5 negativas** (Gráfico 6 — scatter `volatile acidity` vs `quality`):
1. Acidez volátil (-0,41)
2. Densidade (-0,19)
3. Dióxido de enxofre total (-0,18)
4. Cloretos (-0,12)
5. Dióxido de enxofre livre (-0,07)

### 3.8 Análise de distribuições por cluster

**Gráfico de KDE** (density plots) para todas as variáveis, separadas por `cluster_quality`:
- `alcohol` apresenta diferença nítida entre os clusters: vinhos de alta qualidade concentram-se em teores alcoólicos mais altos (~11-12%)
- `volatile acidity` mostra o inverso: alta qualidade concentra-se em valores mais baixos
- `sulphates` e `citric acid` também apresentam separação visível

**Decisão**: a diferença de distribuições entre os clusters confirma que as variáveis com maior separação são fortes candidatas a preditoras.

**Boxplots por cluster** para análise de outliers:
- Presença significativa de outliers em `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`
- **Decisão**: optou-se pela **padronização (StandardScaler)** em vez de normalização Min-Max, por ser mais robusta a outliers

### 3.9 Pairplot

**Pairplot** de todas as variáveis com hue por `cluster_quality`:
- Relações bivariadas entre álcool e densidade mostram padrão de dependência linear inversa
- `pH` e `fixed acidity` apresentam relação química esperada (ácido fraco)
- A separação visual entre clusters é mais evidente nos pares que envolvem `alcohol`, `volatile acidity` e `sulphates`

### 3.10 Scatter plots de relações químicas

Scatter plots específicos investigaram relações entre variáveis quimicamente vinculadas:
- **Álcool vs Densidade**: relação inversa clara — fermentação converte açúcar em álcool, reduzindo a densidade. Vinhos de alta qualidade tendem a ter maior álcool e menor densidade.
- **Cloretos vs Densidade**: correlação positiva, sem separação expressiva entre clusters
- **Acidez fixa vs pH**: relação inversa esperada
- **Acidez fixa vs Densidade**: relação positiva, consistente com a química de soluções

### 3.11 Pesquisa externa

Foi consultado o artigo da **Carnegie Mellon University** (CMU, 2024) sobre predição de qualidade de vinhos ([link](https://www.stat.cmu.edu/capstoneresearch/fall2024/315files_f24/team8.html)), que corroborou os achados:
- Álcool é o preditor mais forte de qualidade
- Densidade e álcool são quimicamente dependentes (multicolinearidade)
- Random Forest é robusto a essa dependência pela aleatorização das variáveis

### 3.12 Decisões de pré-processamento

Com base na EDA, as seguintes decisões foram tomadas:
- **Padronização (StandardScaler)**: escolhida por haver outliers significativos na base
- **Pré-processamento por modelo**: será aplicado individualmente em cada notebook de modelagem, pois cada algoritmo tem requisitos específicos:
  - **Regressão Logística**: requer padronização (sensível à escala)
  - **Random Forest**: não requer padronização (baseado em árvores, robusto a escalas)
  - **XGBoost**: não requer padronização (baseado em árvores, robusto a escalas)

### 3.13 Insight sobre multicolinearidade

Teor alcoólico e densidade apresentam **dependência química**: durante a fermentação, o açúcar se transforma em álcool, reduzindo naturalmente a densidade do líquido. Essa relação foi confirmada pelo heatmap de correlação, pelo pairplot, e pelo artigo da CMU. Modelos baseados em árvores (Random Forest, XGBoost) lidam bem com essa dependência devido à aleatorização das variáveis em cada nó de decisão.

---

## 4. Pré-processamento

Pipeline aplicado a todos os modelos:

1. **Remoção de duplicatas** (já realizada na EDA)
2. **Separação treino-teste**: 70/30 com estratificação (`random_state=42`)
3. **Padronização** (StandardScaler) — aplicada apenas na Regressão Logística
4. **Balanceamento**: SMOTE aplicado na base de treino para Regressão Logística e XGBoost
5. **Threshold de decisão**: calibrado para **0,35** (priorizando recall da classe minoritária)

---

## 5. Modelos

### 5.1 Regressão Logística

**Arquivo**: `models/logisticregression.ipynb`  
**Responsável**: Maycon Suel da Silva Nunes

#### Pipeline

1. Carregamento da base limpa (`data/base_wine_limpa.csv`, 1018 registros)
2. Separação X (11 features físico-químicas) e y (`cluster_quality`)
3. **Divisão treino-teste**: 70/30 com estratificação (`random_state=42`) → 712 treino, 306 teste
4. **StandardScaler**: `fit_transform` no treino, `transform` no teste (regra de ouro: nunca aprender parâmetros do teste)
5. **SMOTE**: balanceamento da base de treino (616 × 616) — aplicado após a divisão, somente no treino

#### Modelo base

`LogisticRegression(max_iter=1000, random_state=42)` com threshold padrão (0,50):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Comum (0) | 0,97 | 0,81 | 0,88 |
| Alta (1) | 0,40 | 0,83 | 0,54 |

Acurácia: 0,81

#### Otimização com GridSearchCV

- Grade: `C = [0.01, 0.1, 1, 10]`, `solver = ['lbfgs', 'liblinear']`
- Validação cruzada: 5 folds
- Métrica de otimização: **F1-Score** (classe minoritária)
- **Melhores parâmetros**: `C = 0.1`, `solver = 'lbfgs'`
- **Melhor F1 médio (CV)**: 0,8447

Modelo otimizado (threshold 0,50): AUC 0,9015 — mesmas métricas do base (GridSearch não alterou o F1 no teste).

#### Ajuste de threshold (0,35)

Reduzindo o ponto de corte de 0,50 para 0,35, o modelo prioriza recall da classe minoritária:

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Comum (0) | 0,98 | 0,74 | 0,84 |
| Alta (1) | 0,35 | 0,90 | 0,50 |

Acurácia: 0,76 | AUC: 0,9015

#### Visualizações (3 gráficos combinados)
- **Matriz de confusão** (threshold 0,35): mostra distribuição de acertos e erros por classe
- **Curva ROC**: AUC de 0,9015, confirmando boa capacidade discriminativa
- **Coeficientes do modelo**: gráfico de barras horizontais onde:
  - Vermelho = coeficiente positivo (aumenta chance de alta qualidade)
  - Azul = coeficiente negativo (reduz chance de alta qualidade)
  - Variáveis mais influentes: **álcool** (+), **acidez volátil** (-), **sulfatos** (+)

#### Evolução das métricas (classe Alta Qualidade)

| Etapa | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Modelo base | 0,50 | 0,40 | 0,83 | 0,54 |
| GridSearch otimizado | 0,50 | 0,40 | 0,83 | 0,54 |
| Threshold ajustado | **0,35** | 0,35 | **0,90** | 0,50 |

**Decisão**: o threshold 0,35 foi escolhido como versão final do modelo, priorizando Recall (identificar 90% dos vinhos de alta qualidade) mesmo com redução de Precision.

#### Exportação
Resultados salvos em `data/resultados_logistica.csv` (y_real, y_pred, y_prob).

---

### 5.2 Random Forest

**Arquivo**: `models/randomforest.ipynb`  
**Responsável**: Carlos Henrique Freitas

#### Pipeline

1. Carregamento da base limpa (1018 registros)
2. Separação X (11 features) e y (`cluster_quality`)
3. **Divisão treino-teste**: 70/30 com estratificação (`random_state=42`)

> O Random Forest não requer padronização ou normalização por ser baseado em árvores de decisão, que são robustas a diferenças de escala entre as variáveis.

#### Modelo original (baseline)

`RandomForestClassifier(n_estimators=100, random_state=42)` com threshold 0,50:

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Comum (0) | 0,91 | 0,98 | 0,94 |
| Alta (1) | 0,73 | 0,39 | 0,51 |

- Acurácia: 0,90
- Matriz de confusão: `[[259, 6], [25, 16]]` — 25 dos 41 vinhos bons foram classificados como comuns

**Diagnóstico**: AUC de **0,90** (Curva ROC) revelou excelente capacidade discriminativa. O baixo Recall (0,39) não é fraqueza do algoritmo, mas sim rigidez do threshold padrão de 0,50 em cenário desbalanceado.

#### GridSearchCV por AUC

- Grade: `n_estimators=[100, 200, 500]`, `max_leaf_nodes=[16, 32, 64, None]`, `max_depth=[5, 10, None]`
- Validação cruzada: 5 folds
- Métrica: **roc_auc**
- **Melhores parâmetros**: `max_depth=5`, `max_leaf_nodes=None`, `n_estimators=500`
- **Melhor AUC médio (CV)**: 0,87
- Resultado no teste: Precision 0,77, Recall **0,24**, F1 **0,37** — pior que o baseline!

**Conclusão**: limitar `max_depth` para evitar overfitting prejudicou a capacidade de capturar padrões da classe minoritária.

#### GridSearchCV por F1

- Mesma grade, métrica: **f1**
- **Melhores parâmetros**: `max_depth=None`, `max_leaf_nodes=None`, `n_estimators=100`
- Resultado: **idêntico ao baseline original** em todos os thresholds testados

**Conclusão**: a otimização por F1 convergiu para a configuração original (árvores sem restrição), validando que o baseline já era a estrutura ótima.

#### Ajuste de threshold — experimento completo

Teste de 7 thresholds no modelo original:

| Threshold | Precision | Recall | F1-Score |
|---|---|---|---|
| 0,20 | 0,43 | 0,80 | 0,56 |
| 0,25 | 0,46 | 0,78 | 0,58 |
| 0,30 | 0,55 | 0,73 | 0,62 |
| **0,35** | **0,66** | **0,71** | **0,68** |
| 0,40 | 0,70 | 0,56 | 0,62 |
| 0,45 | 0,72 | 0,51 | 0,60 |
| 0,50 | 0,74 | 0,41 | 0,53 |

**Threshold 0,35** foi o ponto ótimo, maximizando o F1-Score (0,68) com equilíbrio entre Precision (0,66) e Recall (0,71).

#### Modelo final

`RandomForestClassifier(n_estimators=100, random_state=42)` com threshold **0,35**:

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Comum (0) | 0,95 | 0,94 | 0,95 |
| Alta (1) | 0,66 | 0,71 | 0,68 |

Acurácia: 0,91 | AUC: 0,9049

#### Principais insights

1. **Threshold tuning foi mais eficaz que GridSearch**: calibrar o ponto de corte gerou ganhos maiores que a busca exaustiva de hiperparâmetros
2. **AUC manteve-se estável (~0,90)** em todas as configurações, confirmando que a capacidade discriminativa do Random Forest independe da tunagem
3. **GridSearch por AUC escolheu árvores restritas (max_depth=5)** que pioraram o Recall; já a busca por F1 convergiu para árvores sem restrição
4. **Feature importance nativa** do Random Forest identificou álcool, acidez volátil e sulfatos como variáveis mais relevantes

---

### 5.3 XGBoost

**Arquivo**: `models/xgboost.ipynb`  
**Responsável**: Vinicius Lopes Romão

#### Pipeline

1. Carregamento da base limpa (1018 registros)
2. Remoção da coluna `quality` (evitar data leakage)
3. Separação X (11 features) e y (`cluster_quality`)
4. **Divisão treino-teste**: 70/30 com estratificação (`random_state=42`)

#### Baseline

`XGBClassifier(objective='binary:logistic')`:

- **Baseline simples** (prever tudo como classe 1): 13,46%
- **Baseline Dummy** (prever tudo como classe 0): 86,60%
- Acurácia do modelo: **90,85%**

Relatório de classificação (threshold 0,50):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,93 | 0,97 | 0,95 |
| 1 | 0,72 | 0,51 | 0,60 |

**Diagnóstico**: matriz de confusão mostrou que **57% dos vinhos de alta qualidade** foram classificados como comuns. Recall de 0,51 indica que o modelo ignora metade da classe minoritária.

#### Estratégia 1: Ajuste de threshold (0,30)

Teste com threshold reduzido para 0,30:

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,94 | 0,95 | 0,94 |
| 1 | 0,63 | 0,59 | 0,61 |

Recall subiu de 0,51 para 0,59, mas Precision caiu de 0,72 para 0,63.

#### Estratégia 2: Scale Pos Weight

Balanceamento via peso das classes (`scale_pos_weight` = proporção 616/96 ≈ 6,42):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,94 | 0,94 | 0,94 |
| 1 | 0,61 | 0,61 | 0,61 |

Recall e Precision equilibrados em 0,61, F1 de 0,61.

#### Estratégia 3: SMOTE

Balanceamento sintético da base de treino (616 × 616):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,95 | 0,90 | 0,92 |
| 1 | 0,51 | 0,68 | 0,58 |

Recall subiu para 0,68 (melhor até aqui), mas Precision caiu para 0,51.

#### Estratégia 4: RandomizedSearchCV

Combinação de SMOTE + busca aleatória de hiperparâmetros:

- Grade: `max_depth=[3,4,5,6,8]`, `learning_rate=[0.01,0.05,0.1,0.2]`, `n_estimators=[100,200,300,500]`, `subsample=[0.6,0.8,1.0]`, `colsample_bytree=[0.6,0.8,1.0]`
- 20 iterações, validação cruzada de 3 folds
- Métrica de otimização: **F1-Score**
- **Melhores parâmetros**: `subsample=0.8`, `n_estimators=500`, `max_depth=6`, `learning_rate=0.2`, `colsample_bytree=1.0`

Resultado (threshold 0,50):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,94 | 0,90 | 0,92 |
| 1 | 0,50 | 0,63 | 0,56 |

#### Modelo final: RandomizedSearch + threshold 0,35

Aplicando o threshold de 0,35 (menos que o testado de 0,30 para maximizar recall sem perder muita precisão):

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 | 0,94 | 0,88 | 0,91 |
| 1 | 0,46 | 0,63 | 0,53 |

Acurácia: 0,85 | AUC: 0,9046

Matriz de confusão: `[[234, 31], [15, 26]]`

#### Síntese das estratégias

| Estratégia | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (threshold 0,50) | 0,72 | 0,51 | 0,60 |
| Threshold 0,30 | 0,63 | 0,59 | 0,61 |
| Scale Pos Weight | 0,61 | 0,61 | 0,61 |
| SMOTE | 0,51 | 0,68 | 0,58 |
| RandomizedSearch + SMOTE | 0,50 | 0,63 | 0,56 |
| **RandomizedSearch + threshold 0,35** | **0,46** | **0,63** | **0,53** |

**Decisão**: o modelo final foi o RandomizedSearchCV com threshold 0,35, priorizando Recall de 0,63 (identificar ~26 dos 41 vinhos de alta qualidade) com AUC de 0,9046.

#### Exportação
Resultados salvos em `data/resultados_xgboost.csv`.

---

## 6. Comparação Final

Threshold = 0,35 para todos os modelos.

| Modelo | Precision | Recall | F1-Score | Acurácia | AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,35 | 0,90 | 0,50 | 0,76 | 0,9015 |
| **Random Forest** | **0,66** | **0,71** | **0,68** | **0,91** | **0,9049** |
| XGBoost | 0,46 | 0,63 | 0,53 | 0,85 | 0,9046 |

---

## 7. Modelo Selecionado: Random Forest

O Random Forest foi escolhido como modelo final pelos seguintes critérios:

- **Melhor F1-Score (0,68)**: melhor equilíbrio entre precision e recall para a classe minoritária
- **Maior Acurácia (0,91)**: desempenho geral superior
- **Melhor AUC (0,9049)**: melhor capacidade de discriminação entre as classes
- **Precision mais alta (0,66)**: quando o modelo prevê "alta qualidade", acerta em 66% dos casos — 2× mais que a Regressão Logística (0,35)
- **Feature importance nativa**: permite interpretar quais variáveis mais influenciam a decisão

### Importância das Features (Random Forest)

1. **Teor alcoólico** — variável mais relevante
2. **Acidez volátil** — segunda maior importância
3. **Sulfatos**
4. **Açúcar residual**
5. **Densidade**

---

## 8. Conclusões

1. **Threshold foi mais impactante que hiperparâmetros**: calibrar o ponto de corte para 0,35 gerou ganhos maiores que a busca exaustiva de parâmetros, especialmente para recall da classe minoritária.

2. **AUC similar entre modelos (~0,90)**: os três modelos apresentaram capacidade de discriminação equivalente, mas métricas calibradas (F1, Precision, Recall) mostraram diferenças significativas na prática.

3. **Random Forest foi o modelo mais robusto**: combinou a melhor acurácia (0,91) com o melhor F1 (0,68), indicando generalização superior sem sacrificar a detecção de vinhos de alta qualidade.

4. **Teor alcoólico e acidez volátil são os principais preditores**: consistente com a literatura de enologia — vinhos de maior qualidade tendem a ter maior teor alcoólico e menor acidez volátil.

5. **A multicolinearidade entre álcool e densidade não prejudicou os modelos ensemble**, que lidam naturalmente com essa característica dos dados.

---

## 9. Estrutura do projeto

```
PosGraduacao_DataAnalytics---Machine-Learning---Grupo-31/
│
├── data/
│   ├── WineQT.csv                    # Dataset original
│   ├── base_wine_limpa.csv           # Após limpeza (EDA)
│   ├── resultados_logistica.csv      # Predições Regressão Logística
│   ├── resultados_random_forest.csv  # Predições Random Forest
│   └── resultados_xgboost.csv        # Predições XGBoost
│
├── notebooks/
│   ├── Analise_Base_Vinhos.ipynb     # EDA completa
│   └── Comparacao_Resultados.ipynb   # Comparação final dos modelos
│
├── models/
│   ├── logisticregression.ipynb      # Regressão Logística + SMOTE + GridSearch
│   ├── randomforest.ipynb            # Random Forest + GridSearch + Threshold
│   └── xgboost.ipynb                 # XGBoost + SMOTE + RandomizedSearch
│
├── results/                          # Gráficos e artefatos gerados
├── requirements.txt                  # Dependências
└── README.md                         # Este documento
```

---

## 10. Equipe — Grupo 31

| Nome | GitHub |
|---|---|
| Carlos Henrique Freitas | [Finnagun](https://github.com/Finnagun) |
| Vinicius Lopes Romão | [viniromao159](https://github.com/viniromao159) |
| Maycon Suel da Silva Nunes | [MayconNune](https://github.com/MayconNune) |

---

**Curso**: Pós-Graduação em Data Analytics — POSTECH  
**Fase**: Tech Challenge — Fase 2  
**Grupo**: 31

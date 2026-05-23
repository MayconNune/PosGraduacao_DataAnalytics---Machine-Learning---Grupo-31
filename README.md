# 🍷 Descobrindo a Qualidade de Vinhos com Machine Learning

> Projeto desenvolvido para o Tech Challenge — Fase 2 da Pós-Graduação em Data Analytics (POSTECH)

---

## 🧠 O desafio

Avaliar a qualidade de um vinho normalmente depende da experiência de especialistas, que analisam diversos fatores para chegar a uma nota final. Mas surge uma pergunta:

**Será que um modelo de Machine Learning consegue aprender esses padrões e prever a qualidade de um vinho automaticamente?**

Neste projeto utilizamos características físico-químicas dos vinhos — como acidez, teor alcoólico, pH, sulfatos e densidade — para treinar modelos capazes de identificar se um vinho possui **alta qualidade** ou **baixa/média qualidade**.

Transformamos as notas originais do dataset em uma classificação binária:

✅ Alta Qualidade → nota ≥ 7  
❌ Baixa/Média Qualidade → nota < 7

---

## 🎯 Objetivo

Desenvolver, treinar e comparar diferentes modelos de Machine Learning para descobrir:

- Qual modelo apresenta o melhor desempenho
- Quais características mais influenciam na qualidade do vinho
- Como interpretar os resultados de forma simples e objetiva

---

## 📂 Estrutura do projeto

```bash
PosGraduacao_DataAnalytics---Machine-Learning---Grupo-31/
│
├── data/                # Base de dados utilizada
├── notebooks/           # Exploração, análises e modelagem
├── src/                 # Scripts auxiliares
├── results/             # Resultados, métricas e gráficos
├── requirements.txt     # Dependências do projeto
└── README.md
```

---

## 🔍 Dataset utilizado

**Wine Quality Dataset**

O conjunto de dados contém informações físico-químicas dos vinhos, como:

🍇 Acidez fixa  
🍇 Acidez volátil  
🍇 Ácido cítrico  
🍇 Açúcar residual  
🍇 Cloretos  
🍇 Dióxido de enxofre livre e total  
🍇 Densidade  
🍇 pH  
🍇 Sulfatos  
🍇 Teor alcoólico  
🍇 Qualidade

Fonte:

https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

---

## 🤖 Modelos utilizados

Testamos abordagens diferentes, começando de modelos mais simples até modelos mais sofisticados.

### 📈 Regressão Logística

Modelo utilizado como base de comparação.

**Por que usar?**
- Fácil interpretação
- Rápido para treinar
- Permite entender o impacto das variáveis

---

### 🌳 Random Forest

Uma "floresta" formada por várias árvores de decisão trabalhando juntas.

**Por que usar?**

- Boa performance
- Resistente a ruídos
- Gera importância das variáveis automaticamente

---

### 🚀 XGBoost

Modelo focado em melhorar seus próprios erros a cada etapa.

**Por que usar?**

- Excelente desempenho
- Muito utilizado no mercado
- Ótimo para problemas de classificação

---

## 🔄 Pipeline do projeto

Nosso fluxo de trabalho seguiu as seguintes etapas:

1️⃣ Entendimento do problema

2️⃣ Análise exploratória dos dados (EDA)

3️⃣ Pré-processamento

4️⃣ Treinamento dos modelos

5️⃣ Avaliação de desempenho

6️⃣ Interpretação dos resultados

---

## 📊 Métricas utilizadas

Como apenas acurácia pode ser enganosa, utilizamos múltiplas métricas:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matriz de Confusão

---

## 📌 Principais perguntas que queremos responder

- O álcool realmente influencia na qualidade?
- A acidez tem impacto relevante?
- Quais variáveis possuem maior peso na classificação?
- O modelo consegue identificar vinhos de alta qualidade de forma confiável?

---

## 👥 Integrantes — Grupo 31

| Nome | GitHub |
|---|---|
| Carlos Henrique Freitas | — |
| Vinicius Lopes Romão | — |
| Maycon Suel da Silva Nunes | https://github.com/MayconNune |

---

## 🎓 Sobre o projeto

**Curso:** Pós-Graduação em Data Analytics  
**Instituição:** POSTECH  
**Fase:** Tech Challenge — Fase 2  
**Grupo:** 31

---

### "Entre química e Machine Learning, tentamos descobrir o que faz um vinho ser realmente bom." 🍷

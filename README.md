🍷 Classificação de Qualidade de Vinhos com Machine Learning

Projeto desenvolvido como parte do Tech Challenge da Fase 2 da Pós-Graduação em Data Analytics — POSTECH


📌 Sobre o Projeto
A avaliação de vinhos tradicionalmente é feita por especialistas, o que pode ser um processo demorado e subjetivo. A ideia deste projeto é usar dados físico-químicos dos vinhos — como acidez, teor alcoólico e pH — para treinar modelos de Machine Learning que consigam prever automaticamente se um vinho é de alta qualidade ou não.
Para isso, usamos o Wine Quality Dataset do Kaggle, onde cada amostra de vinho já tem uma nota dada por especialistas. Transformamos essa nota em uma classificação simples:

✅ Alta Qualidade → nota ≥ 7
❌ Baixa/Média Qualidade → nota < 7

Treinamos e comparamos 3 modelos: Regressão Logística, Random Forest e XGBoost, avaliando qual deles performa melhor nessa tarefa.

📂 Estrutura do Repositório
PosGraduacao_DataAnalytics---Machine-Learning---Grupo-31/
│
├── data/               # Base de dados utilizada
├── notebooks/          # Notebooks com análise e modelagem
├── src/                # Scripts auxiliares
├── results/            # Gráficos e métricas dos modelos
├── requirements.txt    # Bibliotecas utilizadas
└── README.md           # Descrição do projeto

🤖 Modelos Utilizados
ModeloResponsávelRegressão LogísticaA definirRandom ForestA definirXGBoostA definir

📊 Dataset

Fonte: Wine Quality Dataset — Kaggle
Variáveis: Acidez fixa, acidez volátil, ácido cítrico, açúcar residual, cloretos, dióxido de enxofre livre, dióxido de enxofre total, densidade, pH, sulfatos, teor alcoólico e qualidade


🔄 Pipeline do Projeto

• Compreensão do problema e definição da variável alvo
• Análise Exploratória de Dados (EDA)
• Pré-processamento dos dados
• Desenvolvimento e treinamento dos modelos
• Avaliação e comparação dos modelos
• Interpretação dos resultados


👥 Integrantes — Grupo 31
• Carlos Henrique Freitas -
• Vinicius Lopes Romão -
• Maycon Suel da Silva -

🎓 Informações do Curso

Curso: Pós-Graduação em Data Analytics
Fase: 2 — Machine Learning
Instituição: FIAP

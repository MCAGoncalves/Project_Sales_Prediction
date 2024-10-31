# Project_Sales_Prediction
## Objetivo do Projeto
Este projeto foi desenvolvido para prever o valor de venda de projetos de uma empresa de consultoria no sul do Brasil. A empresa trabalha com uma variedade de serviços, incluindo estratégia, produção, qualidade e inovação, e enfrenta desafios para estimar com precisão os preços de novos projetos devido à diversidade de variáveis envolvidas.

## Resumo
Devido à diversidade de tipos de projetos e variáveis preditoras (como tipo de consultor, tipo de projeto e número de horas), há uma necessidade de uma análise multivariada robusta para prever os valores de vendas com alta precisão. O projeto aplicou técnicas de machine learning e métodos de Planejamento e Controle da Produção (PPC) para desenvolver um modelo preditivo. Entre os algoritmos testados, o Gradient Boosting Machine (GBM) obteve a menor taxa de erro (aproximadamente 22%), atendendo às expectativas dos stakeholders.

## Etapas do Projeto
- Revisão de literatura sobre PPC e técnicas de Machine Learning.
- Mapeamento do processo atual de prospecção de vendas da empresa.
- Coleta, análise e preparação de dados.
- Teste e comparação de modelos preditivos para seleção do melhor desempenho.
- Discussão dos resultados e proposta de melhorias junto à organização.

## Tecnologias e Bibliotecas
- Python
- Pandas, Scikit-Learn, Numpy, Seaborn, Matplotlib

## Código
O código realiza:
1. Carregamento e limpeza de dados.
2. Criação e treino de modelos de regressão supervisionada.
3. Cálculo de métricas de erro para avaliar o desempenho dos modelos.
4. Visualização dos resultados e comparação com o conjunto de teste.

## Resultados
- Modelo final: Gradient Boosting Machine (GBM) com taxa de erro de 22%.
- A implementação do modelo demonstra o potencial de algoritmos computacionais para previsão de demanda e precificação de projetos na empresa.

## Estrutura do Projeto
- `scripts/`: Contém o script principal do projeto (`project_sales_prediction.py`).
- `data/`: Instruções de acesso ao dataset, caso seja necessário.
- `results/`: Outputs e gráficos dos resultados.

  ## Artigo publicado
- Para acesso ao artigo publicado, segue o DOI https://dx.doi.org/10.22456/1983-8026.127422

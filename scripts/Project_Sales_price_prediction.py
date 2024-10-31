```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Carregando o conjunto de dados
database = pd.read_excel('https://github.com/MCAGoncalves/dataset/raw/main/Project_Salesv1.xlsx')

# Separando as variáveis independentes (X) da variável dependente (y)
X = database.drop(['Valor'], axis=1)
y = database['Valor']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True, test_size=0.3)

# Lista dos modelos de machine learning a serem testados
models = [
    GradientBoostingRegressor(learning_rate=0.1, n_estimators=100),
    KNeighborsRegressor(n_neighbors=20),
    SVR(),
    DecisionTreeRegressor(random_state=1),
    LinearRegression()
]

# Listas para armazenar previsões e métricas de erro para cada modelo
predictions_list = []
MAE_list = []
MAPE_list = []
MSE_list = []
RMSE_list = []

# Loop através dos modelos, treinando e calculando previsões e erros
for model in models:
    # Treinando o modelo
    model.fit(X_train, y_train)
    # Realizando a previsão com o conjunto de teste
    y_pred = model.predict(X_test)
    predictions_list.append(y_pred)
    # Calculando métricas de erro
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    MAE_list.append(MAE)
    MAPE_list.append(MAPE)
    MSE_list.append(MSE)
    RMSE_list.append(RMSE)

# Criação de um DataFrame para armazenar as previsões de cada modelo
predictions_df = pd.DataFrame({
    'GB': predictions_list[0],
    'KNN': predictions_list[1],
    'SVM': predictions_list[2],
    'RF': predictions_list[3],
    'LR': predictions_list[4],
})
predictions_df = predictions_df.round(2)
print(predictions_df)

# Criação de um DataFrame para armazenar as métricas de erro
errors_df = pd.DataFrame({
    'MAE': MAE_list,
    'MAPE': MAPE_list,
    'MSE': MSE_list,
    'RMSE': RMSE_list
}, index=['Gradient Boosting', 'KNN', 'SVM', 'Random Forest', 'Linear Regression'])
errors_df = errors_df.round(2)
print(errors_df)

# Encontrando o índice do modelo com menor MAPE (melhor modelo)
best_model_index = np.argmin(MAPE_list)

# Plotando as previsões do melhor modelo comparadas ao conjunto de teste
y_pred_best = predictions_list[best_model_index]
y_test_best = y_test.values

plt.figure(figsize=(10, 6))
plt.plot(y_pred_best, color='blue', linestyle='--', label='Prediction')
plt.plot(y_test_best, color='orange', label='Test Set')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'Comparação entre o melhor modelo ({type(models[best_model_index]).__name__}) e o conjunto de teste')
plt.show()

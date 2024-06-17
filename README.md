# Analise-de-dados-de-precos-de-casas
# Exemplo de análise de preços de casas com machine learning 
Esse sistema implementa um conjunto abrangente de técnicas de análise de dados e machine learning para explorar e modelar um conjunto de dados de imóveis. O código abrange desde a exploração inicial de dados até a construção e avaliação de modelos preditivos.

### Bibliotecas Utilizadas
- [requirements.txt](src/requirements.txt): `pip install -r requirements.txt`

## Organização
- [src](src): Nesse diretório estão os scripts do sistema e o arquivo csv, sendo o principal [Analise de dados](src/AnaliseDados.ipynb.ipynb) e o arquivo csv para análise [train.csv](src/train.csv). O código carrega um arquivo CSV chamado "train.csv" e o armazena na variável treino para usar o dataframe.
- Exemplo de utilização para identificar correlação com a coluna-chave:
  
              ```python
              correlations = treino[num_cols].corr()['SalePrice'].sort_values(ascending=False)

              top_correlations = correlations.head(11)
              print("Top 10 correlações com SalePrice:\n", top_correlations)

              bottom_correlations = correlations.tail(10)
              print("\nBottom 10 correlações com SalePrice:\n", bottom_correlations)
              ```

- Exemplo de utilização para avaliar se há diferenças significativas na média de SalePrice entre as categorias de cada característica categórica:

            ```
            anova_results = []

            for col in cat_cols:
                categories = treino[col].unique()
                
                if all(treino[treino[col] == category]['SalePrice'].count() > 1 for category in categories):
                    try:
                        anova = f_oneway(*[treino[treino[col] == category]['SalePrice'] for category in categories])
                        anova_results.append((col, anova.statistic, anova.pvalue))
                    except Exception as e:
                        print(f"Erro ao calcular ANOVA para {col}: {e}")
            
            df_anova = pd.DataFrame(anova_results, columns=['Variável Categórica', 'F-statistic', 'p-value'])
            
            df_anova.sort_values(by='p-value', ascending=True, inplace=True)
            
            top_10_correlations = df_anova.head(10)
            
            print("As 10 maiores correlações:")
            print(top_10_correlations)
            
            bottom_10_correlations = df_anova.tail(10)
            
            print("\nAs 10 menores correlações:")
            print(bottom_10_correlations)
    
            ```

  

- Exemplo de utilização para codificação de recursos categóricos:

            ```python
              le = OrdinalEncoder()
              for column in cat_cols:
              treino[f"{column}_le"] = le.fit_transform(treino[column].values.reshape(-1, 1))

              for column in cat_cols:
              print(column)
              print(treino[column].unique())
              print(treino[f"{column}_le"].unique())
              print()    
            ```


- Exemplo de utilização de Regressão linear Simples para análise do modelo:

              ```
              X = treino[num_features + cat_features]
              y = treino['SalePrice']
            
              # Pré-processamento para dados numéricos: imputação de valores ausentes e normalização
              num_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                          ])
            
              # Pré-processamento para dados categóricos: imputação de valores ausentes e codificação one-hot
              cat_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                        ])
            
              # Combinar etapas de pré-processamento
              preprocessor = ColumnTransformer(
              transformers=[
                    ('num', num_transformer, num_features),
                    ('cat', cat_transformer, cat_features)
                ])
            
              # Criar um pipeline que inclui pré-processamento e o modelo de regressão linear
              pipeline_reg = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
                  ])
            
              # Dividir os dados em conjunto de treino e teste
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
              # Treinar o modelo
              pipeline_reg.fit(X_train, y_train)
            
              # Fazer previsões
              y_pred_reg = pipeline_reg.predict(X_test)
            
              # Avaliar o desempenho do modelo
              mse_reg = mean_squared_error(y_test, y_pred_reg)
              rmse_reg = np.sqrt(mse_reg)
              print(f"Mean Squared Error (Regressão Linear): {mse_reg}")
              print(f"Root Mean Squared Error (Regressão Linear): {rmse_reg}")

              ```

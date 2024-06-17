# Analise-de-dados-de-precos-de-casas
# Exemplo de análise de preços de casas com machine learning 
Esse sistema implementa um conjunto abrangente de técnicas de análise de dados e machine learning para explorar e modelar um conjunto de dados de imóveis. O código abrange desde a exploração inicial de dados até a construção e avaliação de modelos preditivos.

### Bibliotecas Utilizadas
- [requirements.txt](src/requirements.txt): `pip install -r requirements.txt`

## Organização
- [src](src): Nesse diretório estão os scripts do sistema e o arquivo csv, sendo o principal [Analise de dados](src/AnaliseDados.ipynb.ipynb) e o arquivo csv para análise [train.csv](src/train.csv).
      - Exemplo de utilização para identificar correlação com a coluna-chave:
          ```python
            correlations = treino[num_cols].corr()['SalePrice'].sort_values(ascending=False)

            top_correlations = correlations.head(11)
            print("Top 10 correlações com SalePrice:\n", top_correlations)

            bottom_correlations = correlations.tail(10)
            print("\nBottom 10 correlações com SalePrice:\n", bottom_correlations)
          ```

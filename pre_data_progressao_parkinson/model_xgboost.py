import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor, plot_importance

# Importa as funções de pré-processamento do arquivo pre_processamento_dados.py
from pre_data_progressao_parkinson.pre_processamento_dados import load_data, integrate_data, handle_missing_values, duplicata_rows, create_temporal_features


def build_and_evaluate_model():
    # Carrega e integra os dados
    peptides, proteins, clinical_data, supplemental_data = load_data()
    combined_data = integrate_data(
        peptides, proteins, clinical_data, supplemental_data)

    # Aplica as etapas de pré-processamento:
    # tratamento de valores ausentes, remoção de duplicatas e criação de features temporais.
    combined_data = handle_missing_values(combined_data)
    combined_data = duplicata_rows(combined_data)
    combined_data = create_temporal_features(combined_data)

    # Aqui, além dos dados clínicos e das features temporais, as informações de abundância
    # de proteínas (NPX, PeptideAbundance) já estão integradas na base.

    # Seleciona a variável alvo: neste exemplo, usamos "updrs_3".
    combined_data = combined_data.dropna(subset=['updrs_3'])
    target = 'updrs_3'

    # Define as features: removendo a variável alvo e identificadores irrelevantes (ex: patient_id)
    X = combined_data.drop(columns=[target, 'patient_id'])
    y = combined_data[target]

    # Define colunas categóricas e numéricas
    categorical_features = ['Peptide', 'UniProt', 'updrs_3_medication']
    numeric_features = [
        col for col in X.columns if col not in categorical_features]

    # Cria o pré-processador: padronização para numéricas e one-hot encoding para categóricas
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Cria o pipeline: pré-processamento + modelo XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Treina o pipeline
    pipeline.fit(X_train, y_train)

    # Obtém o nome das features após o pré-processamento
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out(
    )
    print("Nomes das Features:", feature_names)

    # Realiza predições e avalia o modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Absolute Error(MAE):", mae)
    print("Mean Squared Error(MSE):", mse)
    print("R² Score:", r2)

    # Visualização: Gráfico de Predições vs. Valores Reais com legendas
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Dados Preditos")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(),
             y_test.max()], 'r--', label="Linha Ideal")
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Preditos")
    plt.title("Predições vs. Valores Reais")
    plt.legend()
    plt.show()

    # Visualização: Importância das Features do XGBoost com legenda
    ax = plot_importance(pipeline.named_steps['model'], max_num_features=10)
    plt.title("Importância das Features")
    # Adiciona uma legenda manual, se necessário:
    ax.legend(["Importância"], loc="upper right")
    plt.show()

    #  Otimização de hiperparâmetros usando GridSearchCV
    param_grid = {
        'model__n_estimators': [100, 200],  # n de arvores (estimadores)
        'model__max_depth': [3, 5, 7],  # profundidade maxima da arvore
        'model__learning_rate': [0.01, 0.1, 0.2]  # taxa de aprendizado
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    print("Melhor R² na validação:", grid_search.best_score_)


if __name__ == "__main__":
    build_and_evaluate_model()

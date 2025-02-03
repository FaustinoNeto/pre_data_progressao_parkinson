import os
import pandas as pd
from gera_grafico import plot
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_peptides = pd.read_csv(os.path.join(base_path, "train_peptides.csv"))
    train_proteins = pd.read_csv(os.path.join(base_path, "train_proteins.csv"))
    train_clinical_data = pd.read_csv(
        os.path.join(base_path, "train_clinical_data.csv"))
    supplemental_clinical_data = pd.read_csv(
        os.path.join(base_path, "supplemental_clinical_data.csv"))

    return train_peptides, train_proteins, train_clinical_data, supplemental_clinical_data


def integrate_data(peptides, proteins, clinical_data, supplemental_clinical_data):
    # Junta os dados clínicos (concatenação)
    full_clinical_data = pd.concat(
        [clinical_data, supplemental_clinical_data], ignore_index=True)

    # Merge de proteínas e peptídeos
    protein_peptide = pd.merge(
        proteins, peptides,
        on=['visit_id', 'patient_id', 'UniProt'],
        how='outer',
        suffixes=('_protein', '_peptide')
    )

    # Merge com os dados clínicos completos
    full_data = pd.merge(
        protein_peptide, full_clinical_data,
        on=['visit_id', 'patient_id'],
        how='left'
    )
    full_data.fillna('')  # Preenche valores ausentes com string vazia
    full_data.sort_values(['patient_id', 'visit_month'], inplace=True)
    full_data.drop(columns=['visit_id', 'visit_month_protein',
                            'visit_month'], inplace=True)

    # Renomeia a coluna upd23b_clinical_state_on_medication para updrs_3_medication
    full_data.rename(columns={
                     'upd23b_clinical_state_on_medication': 'updrs_3_medication',
                     'visit_month_peptide': 'visit_month'}, inplace=True)

    print(full_data.info())   # Exibe informações sobre o DataFrame
    return full_data


def handle_missing_values(df):
    # Verificar valores ausentes
    missing_values = df.isnull().sum()

    # Calcular a porcentagem de valores ausentes por coluna
    missing_percentage = (missing_values / len(df)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0]

    # Plotar a porcentagem de valores ausentes por coluna
    missing_percentage.plot(
        kind='bar', title="Porcentagem de valores ausentes por coluna")
    plt.ylabel('Porcentagem de Valores Ausentes')
    plt.show()

    # Contar patient_id com valores ausentes
    missing_patient_id = df['patient_id'].isnull().sum()
    print(f"patient_id com valores ausentes: {missing_patient_id}")

    # Conta o total de pacientes
    total_patients = df['patient_id'].nunique()
    print(f"Total de pacientes: {total_patients}")

    df.fillna({'updrs_3_medication': 'Desconhecido'}, inplace=True)
    df.to_csv('combined_data.csv', index=False)
    return df


def duplicata_rows(df):
    # Verificar se há linhas duplicadas
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    # Remover linhas duplicadas
    df = df.drop_duplicates()
    return df


def create_temporal_features(df):
    df = df.sort_values(['patient_id', 'visit_month'])
    updrs_cols = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
    for col in updrs_cols:
        df[f'{col}_lag1'] = df.groupby('patient_id')[col].shift(1)
    df['months_since_last_visit'] = df.groupby(
        'patient_id')['visit_month'].diff()
    return df


def detect_outliers(df, threshold=3):
    numerical_cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(df[numerical_cols].apply(zscore))
    df["outlier"] = (z_scores > threshold).any(axis=1)

    # calcular a porcentagem  de outliers
    num_outliers = df["outlier"].sum()
    total_rows = len(df)
    outlier_percentage = (num_outliers / total_rows) * 100

    print(f"Porcentagem de outliers: {outlier_percentage:.2f}%")
    return df


def correlation_analysis(df):
    # Selecionar apenas colunas numéricas
    numerical_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numerical_cols].corr()

    # Criar o heatmap da matriz de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False,
                cmap="coolwarm", linewidths=0.5)
    plt.title("Matriz de Correlação das Variáveis")
    plt.show()

def visualize_updrs_scores(df):
    
    # Filtra os dados para pacientes sem medicação
    df = df[df["updrs_3_medication"] == "Off"]

    # Cria subplots
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 20))
    sns.set_style('darkgrid')
    axs = axs.flatten()

    # Plota cada feature UPDRS
    for x, feature in enumerate(["updrs_1", "updrs_2", "updrs_3", "updrs_4"]):
        ax = axs[x]
        sns.boxplot(data=df, x="visit_month", y=feature, ax=ax)
        sns.pointplot(data=df, x="visit_month", y=feature, color="r", errorbar=None, linestyle=":", ax=ax)
        ax.set_title(f"UPDRS Parte {x+1} score mensal de pacientes sem medicação", fontsize=15)
        ax.set_xlabel("Visita Mês")
        ax.set_ylabel("Score")
        ax.legend(['Mean Score'], loc='upper right')

    # Ajusta o layout
    plt.tight_layout()
    plt.show()


def process_and_visualize_correlation(train_df, supp_df):
    
    q3_train_clinical_df = train_df[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']].dropna()

    # Processamento dos dados suplementares
    q3_supp_clinical_df = supp_df.dropna(subset=['upd23b_clinical_state_on_medication', 'updrs_3', 'updrs_4'])[
        ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
    ]

    # Calcula as matrizes de correlação
    q3_train_corr = q3_train_clinical_df.corr()
    q3_supp_corr = q3_supp_clinical_df.corr()

    # Visualização do heatmap
    sns.color_palette(sns.diverging_palette(230, 20))
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Máscara para ocultar valores superiores da matriz
    mask = np.zeros_like(q3_train_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Mapa de cores divergente
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        q3_train_corr,
        square=True,
        mask=mask,
        linewidth=2.5,
        vmax=0.4,
        vmin=-0.4,
        cmap=cmap,
        cbar=False,
        ax=ax
    )

    # Configuração dos rótulos
    ax.set_yticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

    ax.spines['top'].set_visible(True)

    # Título
    fig.text(0.97, 1, 'Correlation Heatmap Visualization for Training Dataset',
             fontweight='bold', fontfamily='serif', fontsize=10, ha='right')
    plt.show()

    return q3_train_corr, q3_supp_corr

# --------------------------------------
# Pipeline Principal
# --------------------------------------


def main():
    # Carrega dados
    peptides, proteins, clinical_data, supplemental_data = load_data()

    # Integração
    combined_data = integrate_data(
        peptides, proteins, clinical_data, supplemental_data)

    # Tratamento de valores ausentes
    combined_data = handle_missing_values(combined_data)
    
    # Remover linhas duplicadas
    # combined_data = duplicata_rows(combined_data)

    # Engenharia de Recursos Temporais
    # combined_data = create_temporal_features(combined_data)

    # Detecção de Outliers
    # combined_data = detect_outliers(combined_data)

    # Análise de Correlação
    # correlation_analysis(combined_data)


    # Processar e visualizar correlações
    train_corr, supp_corr = process_and_visualize_correlation(
        clinical_data, supplemental_data
    )
    # Visualize UPDRS scores
    visualize_updrs_scores(combined_data)


if __name__ == "__main__":
    main()
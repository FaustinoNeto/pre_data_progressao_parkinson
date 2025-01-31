import os
import pandas as pd
from gera_grafico import plot_missing_values


# --------------------------------------
# 1. Carregamento e Análise Exploratória
# --------------------------------------

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_peptides = pd.read_csv(os.path.join(base_path, "train_peptides.csv"))
    train_proteins = pd.read_csv(os.path.join(base_path, "train_proteins.csv"))
    train_clinical_data = pd.read_csv(
        os.path.join(base_path, "train_clinical_data.csv"))
    supplemental_clinical_data = pd.read_csv(
        os.path.join(base_path, "supplemental_clinical_data.csv"))

    return train_peptides, train_proteins, train_clinical_data, supplemental_clinical_data

# --------------------------------------
# 2. Integração de Dados com Merge Seguro
# --------------------------------------


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
    return full_data


def handle_missing_values(df):
    # Remover As Colunas visit_id, visit_month_peptide, visit_month_protein
    df = df.drop(
        columns=['visit_id', 'visit_month_peptide', 'visit_month_protein'])

    plot_missing_values(df, "Valores Ausentes no DataFrame")

    # Preenchimento Temporal dos Valores Ausentes
    df['upd23b_clinical_state_on_medication'] = df.groupby(['patient_id', 'visit_month'])[
        'upd23b_clinical_state_on_medication'].ffill().bfill()
    # Se ainda houver Valores Ausentes, Preencher com 'Desconhecido'
    df['upd23b_clinical_state_on_medication'].fillna(
        'Desconhecido', inplace=True)

    df['updrs_4'] = df.groupby('patient_id')['updrs_4'].ffill().bfill()
    df['updrs_4'].fillna(df['updrs_4'].median(), inplace=True)
    df['updrs_4_missing'] = df['updrs_4'].isna().astype(int)

    plot_missing_values(df, "Valores Ausentes no DataFrame Atualizado")
    df.dropna(inplace=True)
    return df


def duplicata_rows(df):
    # Verificar se há linhas duplicadas
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    # Remover linhas duplicadas
    df = df.drop_duplicates()
    return df
# --------------------------------------
# Pipeline Principal
# --------------------------------------


def main():
    # Carrega dados
    peptides, proteins, clinical_data, supplemental_data = load_data()

    # Integração
    combined_data = integrate_data(
        peptides, proteins, clinical_data, supplemental_data)
    print(combined_data.info())
    print(combined_data[['visit_id', 'patient_id', 'visit_month_protein',
          'visit_month_peptide', 'visit_month']].head())

    # Tratamento de valores ausentes
    combined_data = handle_missing_values(combined_data)
    print(combined_data.info())

    # Remover linhas duplicadas
    combined_data = duplicata_rows(combined_data)


if __name__ == "__main__":
    main()

import os
import pandas as pd


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

# --------------------------------------
# Pipeline Principal
# --------------------------------------


def main():
    # Carrega dados
    peptides, proteins, clinical_data, supplemental_data = load_data()

    # Integração
    combined_data = integrate_data(
        peptides, proteins, clinical_data, supplemental_data)

    pd.set_option('display.max_columns', None)
    print(combined_data.head())


if __name__ == "__main__":
    main()

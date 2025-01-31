import os


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

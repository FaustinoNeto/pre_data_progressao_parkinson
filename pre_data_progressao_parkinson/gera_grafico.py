import matplotlib.pyplot as plt


def plot_missing_values(df, title="Porcentagem de valores ausentes por coluna"):
    # Calcula a porcentagem de valores ausentes
    missing_values = df.isnull().mean() * 100
    # Filtra apenas colunas com valores ausentes
    missing_values = missing_values[missing_values > 0]

    # Se não houver valores ausentes, evitar erro
    if missing_values.empty:
        print(f"Nenhum valor ausente encontrado. {title} não será plotado.")
        return

    plt.figure(figsize=(10, 6))
    missing_values.sort_values().plot(kind='bar', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Colunas")
    plt.ylabel("Porcentagem de Valores Ausentes")
    plt.xticks(rotation=45, ha='right')
    plt.show()

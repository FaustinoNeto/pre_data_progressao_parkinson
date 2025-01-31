import matplotlib.pyplot as plt


def plot_missing_values(df, title):
    # Contar valores ausentes para cada coluna e gerar um gráfico de barras com a quantidade de valores ausentes por coluna (em porcentagem)
    missing_values = df.isnull().mean() * 100
    missing_values = missing_values[missing_values > 0]

    # Ajustar o tamanho da figura
    plt.figure(figsize=(12, 6))

    # Gerar o gráfico de barras
    ax = missing_values.plot(kind='bar')

    # Ajustar os rótulos do eixo x para caber todas as palavras
    plt.xticks(rotation=45, ha='right')

    # Adicionar rótulo ao eixo y
    plt.ylabel('% de valores ausentes')

    # Adicionar título
    plt.title(title)

    # Adicionar os valores nas barras
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Mostrar o gráfico
    plt.show()

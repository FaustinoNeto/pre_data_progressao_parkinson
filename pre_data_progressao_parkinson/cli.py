import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para progressão de Parkinson: Execute via CLI ou GUI."
    )
    parser.add_argument(
        '--gui', action='store_true', help="Iniciar a interface gráfica."
    )
    args = parser.parse_args()

    if args.gui:
        # Importa e executa a interface gráfica
        from pre_data_progressao_parkinson.gui import main as run_gui
        run_gui()
    else:
        # Executa o pipeline do modelo XGBoost diretamente
        from pre_data_progressao_parkinson.model_xgboost import build_and_evaluate_model
        build_and_evaluate_model()


if __name__ == "__main__":
    main()

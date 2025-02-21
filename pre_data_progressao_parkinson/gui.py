import tkinter as tk
from tkinter import messagebox
from pre_data_progressao_parkinson.model_xgboost import build_and_evaluate_model


def run_model():
    try:
        build_and_evaluate_model()
        messagebox.showinfo("Sucesso", "Modelo executado com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", str(e))


def main():
    root = tk.Tk()
    root.title("Pipeline de Progressão de Parkinson")
    root.geometry("400x200")

    lbl_info = tk.Label(
        root, text="Clique no botão para executar o pipeline de modelagem.")
    lbl_info.pack(pady=10)

    btn_run = tk.Button(root, text="Executar Modelo", command=run_model)
    btn_run.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()

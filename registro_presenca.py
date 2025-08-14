import csv
from datetime import datetime
import os

def registrar_presenca(nome):
    caminho = os.path.join("relatorios", "presenca.csv")
    horario = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(caminho):
        with open(caminho, "w", newline="") as arquivo:
            writer = csv.writer(arquivo)
            writer.writerow(["Nome", "Horário"])

    with open(caminho, "a", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([nome, horario])
    print(f"📝 Presença registrada: {nome} às {horario}")
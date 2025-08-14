import cv2
import face_recognition
import os
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
import atexit
from datetime import datetime
from registro_presenca import registrar_presenca

var = face_recognition.face_landmarks(face_recognition.load_image_file("imagens/prof_daniel.jpg"))

print(var)

CAMINHO_IMAGENS = "imagens"
CAMINHO_RELATORIOS = "relatorios"
CAMINHO_ENTRADAS = "fotos_entrada"
CAMINHO_DADOS_ROSTOS = os.path.join(CAMINHO_IMAGENS, "dados_rostos.npz")

TEMPO_FECHADURA = 5
TOLERANCIA_RECONHECIMENTO = 0.40
BAUD_RATE = 9600
CAMERA_RECONHECIMENTO = 0
CAMERA_CADASTRO = 1

reconhecido_vezes = {}
lock = threading.Lock()
fechadura_fechada = True
arduino = None
arduino_lock = threading.Lock()

os.makedirs(CAMINHO_IMAGENS, exist_ok=True)
os.makedirs(CAMINHO_RELATORIOS, exist_ok=True)
os.makedirs(CAMINHO_ENTRADAS, exist_ok=True)

def selecionar_porta_arduino():
    portas = list(serial.tools.list_ports.comports())
    if not portas:
        print("❌ Nenhum dispositivo serial encontrado. Conecte o Arduino e tente novamente.")
        return None
    
    print("\n🔌 Portas disponíveis:")
    for i, porta in enumerate(portas):
        print(f"{i + 1}. {porta.device} - {porta.description}")

    if len(portas) == 1:
        print(f"✅ Apenas uma porta encontrada. Usando {portas[0].device}.")
        return portas[0].device

    while True:
        escolha = input("Digite o número da porta que deseja usar: ").strip()
        try:
            indice = int(escolha) - 1
            if 0 <= indice < len(portas):
                return portas[indice].device
            else:
                print("⚠️ Número inválido. Tente novamente.")
        except ValueError:
            print("⚠️ Digite um número válido.")

def conectar_arduino():
    global arduino
    while True:
        porta_escolhida = selecionar_porta_arduino()
        if not porta_escolhida:
            return

        try:
            arduino = serial.Serial(porta_escolhida, BAUD_RATE, timeout=1)
            print(f"✅ Conectado ao Arduino na porta {porta_escolhida}.")
            break
        except serial.SerialException as e:
            print(f"❌ Erro ao conectar ao Arduino: {e}")
            print("Aguarde 5 segundos para tentar novamente...")
            time.sleep(5)

def desligar_rele():
    global arduino
    try:
        if arduino and arduino.is_open:
            arduino.write(b'0')
            print("🔒 Fechadura fechada. Comunicação serial encerrada.")
            arduino.close()
    except Exception as e:
        print(f"Erro ao desligar rele: {e}")

atexit.register(desligar_rele)

def escrever_no_arduino(valor: bytes):
    global arduino
    with arduino_lock:
        try:
            if arduino is None:
                print("⚠️ Arduino está None. Não foi possível enviar comando.")
                return
            if not arduino.is_open:
                print("⚠️ A porta serial não está aberta.")
                return
            arduino.write(valor)
        except Exception as e:
            print(f"Erro ao escrever no Arduino: {e}")

def carregar_rostos():
    if os.path.exists(CAMINHO_DADOS_ROSTOS):
        dados = np.load(CAMINHO_DADOS_ROSTOS, allow_pickle=True)
        return list(dados['rostos']), list(dados['nomes'])
    else:
        return [], []

def salvar_rostos(rostos, nomes):
    np.savez(CAMINHO_DADOS_ROSTOS, rostos=rostos, nomes=nomes)

def abrir_fechadura_manual():
    global fechadura_fechada
    print("🔓 Abrindo fechadura manualmente...")
    escrever_no_arduino(b'1')
    time.sleep(TEMPO_FECHADURA)
    escrever_no_arduino(b'0')
    print("🔒 Fechadura fechada.")
    fechadura_fechada = True

def exibir_cameras_lado_a_lado():
    cam_recon = cv2.VideoCapture(CAMERA_RECONHECIMENTO)
    cam_cadastro = cv2.VideoCapture(CAMERA_CADASTRO)

    if not cam_recon.isOpened() or not cam_cadastro.isOpened():
        print("❌ Erro ao acessar as câmeras. Verifique conexões.")
        return

    print("📸 Posicione o rosto para cadastro (câmera da direita). Pressione 's' para salvar ou 'q' para sair.")

    nome = input("Digite o nome da pessoa: ").strip()
    if not nome:
        print("⚠️ Nome inválido.")
        return

    while True:
        ret_recon, frame_recon = cam_recon.read()
        ret_cad, frame_cadastro = cam_cadastro.read()

        if not ret_recon or frame_recon is None or not ret_cad or frame_cadastro is None:
            print("⚠️ Falha ao capturar imagem de uma das câmeras.")
            continue

        frame_recon = cv2.resize(frame_recon, (320, 240))
        frame_cadastro = cv2.resize(frame_cadastro, (320, 240))
        combinado = np.hstack((frame_recon, frame_cadastro))

        cv2.imshow("Reconhecimento | Cadastro", combinado)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("s"):
            caminho = os.path.join(CAMINHO_IMAGENS, f"{nome}.jpg")
            cv2.imwrite(caminho, frame_cadastro)
            imagem = face_recognition.load_image_file(caminho)
            codificacoes = face_recognition.face_encodings(imagem)
            if codificacoes:
                rosto_codificado = codificacoes[0]
                rostos, nomes = carregar_rostos()
                rostos.append(rosto_codificado)
                nomes.append(nome)
                salvar_rostos(rostos, nomes)
                print(f"✅ {nome} cadastrado com sucesso.")
            else:
                print("❌ Nenhum rosto detectado.")
                os.remove(caminho)
            break

        elif tecla == ord("q"):
            print("❌ Cadastro cancelado.")
            break

    cam_recon.release()
    cam_cadastro.release()
    cv2.destroyAllWindows()

def reconhecimento_loop():
    rostos_reconhecidos, nomes_reconhecidos = carregar_rostos()
    if not rostos_reconhecidos:
        print("⚠️ Nenhum rosto cadastrado.")
        return

    camera = cv2.VideoCapture(CAMERA_RECONHECIMENTO)
    if not camera.isOpened():
        print("❌ Falha ao acessar a câmera de reconhecimento.")
        return

    print("🔍 Reconhecimento facial iniciado. Pressione 'q' para sair, 'm' para abrir fechadura manualmente.")

    global fechadura_fechada

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("⚠️ Falha ao capturar frame.")
            continue

        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        localizacoes = face_recognition.face_locations(imagem_rgb)
        codificacoes = face_recognition.face_encodings(imagem_rgb, localizacoes)

        for codificacao, localizacao in zip(codificacoes, localizacoes):
            distancias = face_recognition.face_distance(rostos_reconhecidos, codificacao)
            indice = np.argmin(distancias) if distancias.size > 0 else -1
            nome = "Desconhecido"

            if indice != -1 and distancias[indice] < TOLERANCIA_RECONHECIMENTO:
                nome = nomes_reconhecidos[indice]

            top, right, bottom, left = localizacao
            cor = (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
            cv2.putText(frame, nome, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

            if nome != "Desconhecido":
                reconhecido_vezes[nome] = reconhecido_vezes.get(nome, 0) + 1
                if reconhecido_vezes[nome] >= 3 and fechadura_fechada:
                    caminho_imagem = os.path.join(CAMINHO_ENTRADAS, f"{nome}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(caminho_imagem, frame)
                    registrar_presenca(nome)
                    escrever_no_arduino(b'1')
                    fechadura_fechada = False
                    time.sleep(TEMPO_FECHADURA)
                    escrever_no_arduino(b'0')
                    fechadura_fechada = True
                    reconhecido_vezes.clear()

        cv2.imshow("Reconhecimento Facial", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            print("👋 Encerrando reconhecimento facial.")
            break
        elif tecla == ord('m'):
            print("⌨️ Abrindo fechadura manualmente via tecla 'm'...")
            abrir_fechadura_manual()

    camera.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("\n--- Menu ---")
        print("1. Cadastrar novo rosto")
        print("2. Iniciar reconhecimento facial")
        print("3. Abrir fechadura manualmente")
        print("4. Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            exibir_cameras_lado_a_lado()
        elif opcao == "2":
            reconhecimento_loop()
        elif opcao == "3":
            abrir_fechadura_manual()
        elif opcao == "4":
            print("👋 Encerrando...")
            break
        else:
            print("❌ Opção inválida.")

if __name__ == "__main__":
    conectar_arduino()
    if arduino is None:
        print("❌ Não foi possível conectar ao Arduino. Encerrando.")
        exit(1)
    menu()
    desligar_rele()

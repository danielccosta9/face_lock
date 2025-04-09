import cv2
import time
import serial
import os
import threading
import atexit
import numpy as np

# Criar pasta de rostos cadastrados, se não existir
def criar_pasta_rostos():
    pasta_rostos = "rostos_cadastrados"
    if not os.path.exists(pasta_rostos):
        os.makedirs(pasta_rostos)
        print(f"Pasta '{pasta_rostos}' criada com sucesso.")
    return pasta_rostos

pasta_rostos = criar_pasta_rostos()
rostos_cadastrados = []
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para testar as câmeras conectadas
def testar_cameras():
    for i in range(4):  # Testa as câmeras nos índices 0, 1, 2, 3
        video_capture = cv2.VideoCapture(i)
        if video_capture.isOpened():
            print(f"Câmera encontrada no índice {i}")
            video_capture.release()  # Libera a câmera depois de testá-la
            return i  # Retorna o índice da câmera encontrada
    print("Nenhuma câmera encontrada.")
    return None  # Se nenhuma câmera foi encontrada

# Testar câmeras conectadas e escolher a primeira que funciona
indice_camera = testar_cameras()
if indice_camera is None:
    print("Erro: Nenhuma câmera disponível.")
    exit(1)  # Se nenhuma câmera for encontrada, encerra o programa

# Carregar as imagens cadastradas e treinar o modelo de reconhecimento
def carregar_rostos_cadastrados():
    global rostos_cadastrados, recognizer
    rostos_cadastrados = []  # Limpa a lista de rostos cadastrados
    imagens = []
    rotulos = []
    nomes_map = {}  # Dicionário para mapear índices para nomes

    # Carregar todas as imagens e seus rótulos
    for indice, arquivo in enumerate(os.listdir(pasta_rostos)):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
            caminho_imagem = os.path.join(pasta_rostos, arquivo)
            nome = os.path.splitext(arquivo)[0]
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            if imagem is not None:
                imagem_rosto = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = imagem_rosto.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    rosto = imagem[y:y+h, x:x+w]
                    imagens.append(rosto)
                    rotulos.append(indice)  # Usando índice numérico como rótulo
                    nomes_map[indice] = nome  # Mapeando o índice para o nome

    if len(imagens) > 0:
        recognizer.train(imagens, np.array(rotulos))  # Treina o modelo com as imagens e seus rótulos
        print(f"{len(imagens)} rostos carregados e modelo treinado.")
    else:
        print("Nenhuma face detectada durante o treinamento.")
    return nomes_map

# Conectar ao Arduino automaticamente
def conectar_arduino():
    global arduino
    while True:
        try:
            arduino = serial.Serial('COM6', 9600, timeout=1)
            print("Conectado ao Arduino.")
            return
        except serial.SerialException:
            print("Erro ao conectar ao Arduino. Tentando novamente...")
            time.sleep(2)

# Monitorar reconexão do Arduino
def monitorar_arduino():
    global arduino
    while True:
        if not arduino.is_open:
            print("Reconectando ao Arduino...")
            conectar_arduino()
        time.sleep(5)

arduino = None
conectar_arduino()

th_monitoramento = threading.Thread(target=monitorar_arduino, daemon=True)
th_monitoramento.start()

# Fechar conexão ao sair do programa
def fechar_arduino():
    if arduino and arduino.is_open:
        arduino.write(b'0')  # Garante que a fechadura está fechada
        arduino.close()
        print("Conexão com Arduino encerrada.")

atexit.register(fechar_arduino)

arduino_lock = threading.Lock()
fechadura_fechada = True

# Cadastrar novo rosto
def cadastrar_rosto():
    nome = input("Digite o nome da pessoa: ").strip()
    if not nome:
        print("Nome inválido. Tente novamente.")
        return

    video_capture = cv2.VideoCapture(indice_camera)  # Usar o índice da câmera testada
    if not video_capture.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione 's' para capturar a imagem ou 'q' para cancelar.")
    while True:
        ret, frame = video_capture.read()
        cv2.imshow("Captura de Rosto", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            caminho_imagem = os.path.join(pasta_rostos, f"{nome}.jpg")
            cv2.imwrite(caminho_imagem, frame)
            print(f"Imagem salva em {caminho_imagem}")
            break
        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    carregar_rostos_cadastrados()  # Atualiza a base de rostos

# Reconhecimento facial otimizado
def reconhecimento_facial():
    global fechadura_fechada, arduino
    video_capture = cv2.VideoCapture(indice_camera)  # Usar o índice da câmera testada
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not video_capture.isOpened():
        print("Erro ao acessar a webcam.")
        return

    tempo_sem_rosto = 0
    reconhecido_vezes = {}

    nomes_map = carregar_rostos_cadastrados()  # Carrega os rostos e treina o modelo uma única vez

    if len(nomes_map) == 0:
        print("Nenhum rosto cadastrado. Por favor, cadastre ao menos um rosto.")
        return

    print("Iniciando reconhecimento facial...")

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_locations = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_locations.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            tempo_sem_rosto += 1
            if tempo_sem_rosto > 30 and not fechadura_fechada:
                with arduino_lock:
                    print("Fechando fechadura por inatividade...")
                    arduino.write(b'0')
                    time.sleep(0.5)
                    fechadura_fechada = True
            continue

        tempo_sem_rosto = 0

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(rosto)
            nome = "Desconhecido"
            if confidence < 100:
                nome = nomes_map.get(label, "Desconhecido")  # Usar o mapeamento de índice para nome

            print(f"Reconhecido: {nome} com confiança {confidence}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if nome != "Desconhecido" and reconhecido_vezes.get(nome, 0) >= 3 and fechadura_fechada:
                with arduino_lock:
                    print(f"Abrindo fechadura para {nome}...")
                    arduino.write(b'1')
                    time.sleep(0.5)
                fechadura_fechada = False

        cv2.imshow('Reconhecimento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Menu
def menu_inicial():
    while True:
        print("\n--- Sistema de Reconhecimento Facial ---")
        print("1 - Cadastrar um novo rosto")
        print("2 - Reconhecer rosto")
        print("3 - Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            cadastrar_rosto()
        elif opcao == "2":
            reconhecimento_facial()
        elif opcao == "3":
            print("Saindo...")
            fechar_arduino()
            break
        else:
            print("Opção inválida. Tente novamente.")

menu_inicial()

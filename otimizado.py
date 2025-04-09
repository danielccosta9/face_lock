import face_recognition
import cv2
import threading
from datetime import datetime
import time

# Constantes
TOLERANCIA_RECONHECIMENTO = 0.4
QUADROS_RECONHECIMENTO = 3

# Caminho das imagens cadastradas
image_paths = [
    {"path": "rostos_cadastrados/hermes.jpg", "name": "Hermes"},
    {"path": "rostos_cadastrados/lindo.jpg", "name": "Alisson"},
    {"path": "rostos_cadastrados/clau.jpg", "name": "Claudio"},
    {"path": "rostos_cadastrados/kaio.jpg", "name": "Kaio"},
    {"path": "rostos_cadastrados/daniel.jpg", "name": "Daniel"},
    {"path": "rostos_cadastrados/everal.jpg", "name": "Everaldio"}
]

# Carregar as imagens cadastradas e suas codificações
cadastrados = [
    {"name": image_info["name"], "encoding": face_recognition.face_encodings(face_recognition.load_image_file(image_info["path"]))[0]}
    for image_info in image_paths
]

# Histórico de reconhecimentos (mantido em memória)
historico_detectados = []

# Lock para o frame
frame_lock = threading.Lock()
frame_atual = None

# Variáveis para controlar o intervalo de prints
ultima_impressao = time.time()
intervalo_impressao = 1  # Em segundos (a cada 1 segundo, vai imprimir o total)

# Função para registrar reconhecimento
def registrar_reconhecimento(nome):
    """Registra o nome da pessoa detectada junto com o horário no histórico."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    historico_detectados.append({"nome": nome, "hora": timestamp})
    print(f"Pessoa detectada: {nome} em {timestamp}")

# Função para exibir o histórico de registros
def exibir_historico():
    """Exibe o histórico de pessoas detectadas."""
    if not historico_detectados:
        print("Nenhuma pessoa detectada ainda.")
    else:
        print("Histórico de Reconhecimentos:")
        for registro in historico_detectados:
            print(f"{registro['nome']} - {registro['hora']}")

# Função para contar quantos estudantes foram detectados
def contar_estudantes():
    """Contabiliza os estudantes detectados e retorna o total."""
    return len(historico_detectados)

# Função de captura de frames da câmera em um thread separado
def captura_video():
    """Captura os frames da câmera em tempo real."""
    global frame_atual
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Erro ao acessar a webcam.")
        return

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Resolução ajustada para melhorar a fluidez
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro ao capturar imagem da webcam.")
            break

        # Atualiza o frame de forma thread-safe
        with frame_lock:
            frame_atual = frame

    video_capture.release()

# Função principal de reconhecimento facial
def reconhecimento_facial():
    """Realiza o reconhecimento facial e registra o histórico."""
    tempo_sem_rosto = 0
    contagem_rostos = {}

    global ultima_impressao

    while True:
        # Usa o frame capturado pelo thread de captura
        with frame_lock:
            if frame_atual is None:
                continue  # Espera até o primeiro frame ser capturado

        frame = frame_atual
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_locations) == 0:
            tempo_sem_rosto += 1
        else:
            tempo_sem_rosto = 0
            for face_encoding, face_location in zip(face_encodings, face_locations):
                nome = "Desconhecido"
                for cadastrado in cadastrados:
                    if face_recognition.compare_faces([cadastrado["encoding"]], face_encoding, tolerance=TOLERANCIA_RECONHECIMENTO)[0]:
                        nome = cadastrado["name"]
                        contagem_rostos[nome] = contagem_rostos.get(nome, 0) + 1

                # Desenhar o rosto detectado na tela
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                cv2.putText(frame, nome, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)

                # Quando o rosto for reconhecido com confiança, registrar
                if contagem_rostos.get(nome, 0) > QUADROS_RECONHECIMENTO:
                    print(f"Rosto reconhecido: {nome} - Registrando no histórico")
                    registrar_reconhecimento(nome)

        # Exibir o frame
        cv2.imshow('Video', frame)

        # Atualizar a contagem de estudantes a cada 1 segundo
        if time.time() - ultima_impressao >= intervalo_impressao:
            ultima_impressao = time.time()
            num_estudantes = contar_estudantes()
            print(f"Estudantes detectados no ônibus: {num_estudantes}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Função para controle manual do histórico
def controle_manual():
    """Permite o controle manual do histórico."""
    while True:
        comando = input("Digite 'historico' para ver o histórico ou 'sair' para encerrar: ").strip().lower()
        if comando == "historico":
            exibir_historico()
        elif comando == "sair":
            print("Saindo do modo manual.")
            break
        else:
            print("Comando inválido. Digite 'historico' ou 'sair'.")

# Função para iniciar os threads
def iniciar_programa():
    """Inicia os threads para reconhecimento facial e controle manual."""
    thread_reconhecimento = threading.Thread(target=reconhecimento_facial)
    thread_controle = threading.Thread(target=controle_manual)
    thread_captura = threading.Thread(target=captura_video)

    # Iniciar os threads
    thread_captura.start()
    thread_reconhecimento.start()
    thread_controle.start()

    thread_captura.join()
    thread_reconhecimento.join()
    thread_controle.join()

# Iniciar o programa
iniciar_programa()

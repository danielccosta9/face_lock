import face_recognition
import cv2
import time
import serial
import atexit
import threading

# Constantes
PORTA_ARDUINO = 'COM6'
BAUD_RATE = 9600
TOLERANCIA_RECONHECIMENTO = 0.4
QUADROS_RECONHECIMENTO = 3
TEMPO_FECHADURA = 5

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

# Variáveis globais
arduino = None
fechadura_fechada = True
arduino_lock = threading.Lock()

def conectar_arduino():
    """Conecta ao Arduino e tenta reconectar em caso de falha."""
    global arduino
    while arduino is None:
        try:
            arduino = serial.Serial(PORTA_ARDUINO, BAUD_RATE, timeout=1)
            print("Conectado ao Arduino.")
        except serial.SerialException as e:
            print(f"Erro ao conectar ao Arduino: {e}. Tentando novamente em 5 segundos...")
            time.sleep(5)

def desligar_rele():
    """Desliga o relé e fecha a comunicação com o Arduino."""
    global arduino
    if arduino and arduino.is_open:
        arduino.write(b'0')  # Envia '0' para desligar o relé
        print("Fechadura fechada e comunicação serial encerrada.")
        arduino.close()

atexit.register(desligar_rele)

def escrever_no_arduino(comando):
    """Função segura para escrever no Arduino."""
    global arduino
    try:
        if arduino and arduino.is_open:
            arduino.write(comando)
        else:
            print("Erro: A porta serial não está aberta.")
    except serial.SerialException as e:
        print(f"Erro ao enviar comando para o Arduino: {e}")

def reconhecimento_facial():
    """Realiza o reconhecimento facial e controla a fechadura."""
    global fechadura_fechada, arduino

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Erro ao acessar a webcam.")
        return

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 300)  
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    tempo_sem_rosto = 0
    contagem_rostos = {}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro ao capturar imagem da webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_locations) == 0:
            tempo_sem_rosto += 1
            if tempo_sem_rosto > 30 and not fechadura_fechada:
                with arduino_lock:
                    if arduino:
                        print("Nenhum rosto detectado. Fechando a fechadura automaticamente.")
                        escrever_no_arduino(b'0')
                        fechadura_fechada = True
        else:
            tempo_sem_rosto = 0
            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Desconhecido"
                
                for cadastrado in cadastrados:
                    if face_recognition.compare_faces([cadastrado["encoding"]], face_encoding, tolerance=TOLERANCIA_RECONHECIMENTO)[0]:
                        name = cadastrado["name"]
                        contagem_rostos[name] = contagem_rostos.get(name, 0) + 1

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.5, (255, 0, 255), 1)

                if contagem_rostos.get(name, 0) > QUADROS_RECONHECIMENTO:
                    with arduino_lock:
                        if arduino:
                            print(f"Rosto reconhecido: {name} - Abrindo a fechadura")
                            escrever_no_arduino(b'1')
                            fechadura_fechada = False
                    time.sleep(TEMPO_FECHADURA)
                    with arduino_lock:
                        if arduino:
                            print("Fechadura fechada após 5 segundos.")
                            escrever_no_arduino(b'0')
                            fechadura_fechada = True
                    break

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def controle_manual():
    """Permite o controle manual da fechadura."""
    global fechadura_fechada

    while True:
        comando = input("Digite 'abrir' para abrir a fechadura ou 'fechar' para fechar: ").strip().lower()
        if comando == "abrir" and fechadura_fechada:
            with arduino_lock:
                if arduino:
                    print("Abrindo a fechadura manualmente.")
                    escrever_no_arduino(b'1')
                    fechadura_fechada = False
                    time.sleep(TEMPO_FECHADURA)
                    escrever_no_arduino(b'0')
                    fechadura_fechada = True
                    print("Fechadura fechada após 5 segundos.")
        elif comando == "fechar" and not fechadura_fechada:
            with arduino_lock:
                if arduino:
                    print("Comando 'fechar' recebido. Fechando a fechadura manualmente.")
                    escrever_no_arduino(b'0')
                    fechadura_fechada = True
        elif comando == "sair":
            print("Saindo do modo manual.")
            break
        else:
            print("Comando inválido. Digite 'abrir' ou 'fechar'.")

def iniciar_programa():
    """Inicia os threads para reconhecimento facial e controle manual."""
    thread_reconhecimento = threading.Thread(target=reconhecimento_facial)
    thread_controle = threading.Thread(target=controle_manual)
    
    thread_reconhecimento.start()
    thread_controle.start()

    thread_reconhecimento.join()
    thread_controle.join()

# Iniciar o programa
conectar_arduino()
iniciar_programa()
desligar_rele()

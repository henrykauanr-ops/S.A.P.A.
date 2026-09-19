import cv2
import numpy as np
import time
import threading
from collections import deque
from playsound import playsound
from ultralytics import YOLO


# ==========================================================
# S.A.P.A. - SISTEMA DE ALERTA E PREVISÃO DE ALAGAMENTO
# Detecção de movimento da água + detecção de garrafas/lixo
# ==========================================================


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

INDICE_CAMERA = 0

# Modelo YOLO usado para detectar garrafas
modelo = YOLO("yolo11n.pt")

# Confiança mínima para considerar uma garrafa
CONFIANCA_MINIMA = 0.80

# Confirmação da garrafa:
# 5 dos últimos 7 frames precisam detectar o objeto
TOTAL_FRAMES = 7
FRAMES_NECESSARIOS = 5

historico_garrafa = deque(maxlen=TOTAL_FRAMES)


# ==========================================================
# REGIÃO DA ÁGUA
# ==========================================================

ROI_X = 100
ROI_Y = 80
ROI_W = 1080
ROI_H = 600

# Limites usados na análise do movimento da água
LIMIAR_MOVIMENTO = 1.0
AREA_MINIMA_AGUA = 5000


# ==========================================================
# ALERTAS
# ==========================================================

INTERVALO_ALERTA = 20

ultimo_alerta_agua = 0
ultimo_alerta_lixo = 0

AUDIO_ALERTA = "ssstik.io_1778940651943.mp3"

audio_tocando = False


def tocar_audio():
    global audio_tocando

    try:
        audio_tocando = True
        playsound(AUDIO_ALERTA)

    except Exception as erro:
        print(f"Erro ao tocar áudio: {erro}")

    finally:
        audio_tocando = False


def alerta_sonoro():
    if not audio_tocando:
        thread_audio = threading.Thread(
            target=tocar_audio,
            daemon=True
        )
        thread_audio.start()


# ==========================================================
# DESENHA DETECÇÃO DA GARRAFA
# ==========================================================

def desenhar_garrafa(imagem, caixa, confianca):

    x1, y1, x2, y2 = map(int, caixa)

    cv2.rectangle(
        imagem,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    texto = f"GARRAFA {confianca * 100:.1f}%"

    cv2.putText(
        imagem,
        texto,
        (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


# ==========================================================
# CÂMERA
# ==========================================================

cap = cv2.VideoCapture(INDICE_CAMERA)

if not cap.isOpened():
    print("ERRO: não foi possível abrir a câmera.")
    exit()

ret, primeiro_frame = cap.read()

if not ret:
    print("ERRO: a câmera abriu, mas não enviou imagem.")
    cap.release()
    exit()


altura, largura = primeiro_frame.shape[:2]

print("======================================")
print("S.A.P.A. INICIADO COM SUCESSO")
print("======================================")
print(f"Resolução: {largura} x {altura}")


# ==========================================================
# AJUSTE DA ROI
# ==========================================================

if ROI_X + ROI_W > largura:
    ROI_W = largura - ROI_X

if ROI_Y + ROI_H > altura:
    ROI_H = altura - ROI_Y

if ROI_W <= 0 or ROI_H <= 0:
    print("ERRO: ROI inválida para a resolução da câmera.")
    cap.release()
    exit()

print(
    f"ROI: X={ROI_X}, Y={ROI_Y}, "
    f"W={ROI_W}, H={ROI_H}"
)


# ==========================================================
# PRIMEIRO FRAME PARA O MOVIMENTO DA ÁGUA
# ==========================================================

roi1 = primeiro_frame[
    ROI_Y:ROI_Y + ROI_H,
    ROI_X:ROI_X + ROI_W
]

prvs = cv2.cvtColor(
    roi1,
    cv2.COLOR_BGR2GRAY
)

prvs = cv2.GaussianBlur(
    prvs,
    (7, 7),
    0
)

historico_movimento = []


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERRO: não foi possível receber imagem da câmera.")
        break


    # ======================================================
    # 1 - DETECÇÃO DE MOVIMENTO DA ÁGUA
    # ======================================================

    roi = frame[
        ROI_Y:ROI_Y + ROI_H,
        ROI_X:ROI_X + ROI_W
    ]

    atual = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    atual = cv2.GaussianBlur(
        atual,
        (7, 7),
        0
    )

    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        atual,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    mag, ang = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    movimento_medio = np.mean(mag)

    historico_movimento.append(movimento_medio)

    if len(historico_movimento) > 30:
        historico_movimento.pop(0)

    media_historica = np.mean(historico_movimento)

    horizontal = np.mean(np.abs(flow[..., 0]))
    vertical = np.mean(np.abs(flow[..., 1]))

    confianca_agua = 0

    if media_historica > LIMIAR_MOVIMENTO:
        confianca_agua += 1

    if horizontal > vertical:
        confianca_agua += 1


    # ======================================================
    # 2 - MÁSCARA DO MOVIMENTO DA ÁGUA
    # ======================================================

    _, movimento = cv2.threshold(
        mag,
        LIMIAR_MOVIMENTO,
        255,
        cv2.THRESH_BINARY
    )

    movimento = movimento.astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    movimento = cv2.morphologyEx(
        movimento,
        cv2.MORPH_OPEN,
        kernel
    )

    movimento = cv2.dilate(
        movimento,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        movimento,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    deteccao_agua = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= AREA_MINIMA_AGUA:
            deteccao_agua = True
            break


    # ======================================================
    # 3 - DETECÇÃO DE GARRAFA COM YOLO
    # ======================================================

    garrafa_detectada = False
    maior_confianca = 0
    melhor_caixa = None

    resultados = modelo(
        frame,
        verbose=False
    )

    for resultado in resultados:

        for caixa in resultado.boxes:

            classe = int(caixa.cls[0])
            confianca = float(caixa.conf[0])
            nome = modelo.names[classe]

            if (
                nome == "bottle"
                and confianca >= CONFIANCA_MINIMA
            ):

                garrafa_detectada = True

                if confianca > maior_confianca:
                    maior_confianca = confianca
                    melhor_caixa = caixa.xyxy[0]


    # ======================================================
    # 4 - SEGUNDA TENTATIVA: IMAGEM GIRADA 90°
    # ======================================================

    if not garrafa_detectada:

        imagem_girada = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

        resultados_girada = modelo(
            imagem_girada,
            verbose=False
        )

        for resultado in resultados_girada:

            for caixa in resultado.boxes:

                classe = int(caixa.cls[0])
                confianca = float(caixa.conf[0])
                nome = modelo.names[classe]

                if (
                    nome == "bottle"
                    and confianca >= CONFIANCA_MINIMA
                ):

                    garrafa_detectada = True

                    if confianca > maior_confianca:

                        maior_confianca = confianca

                        x1, y1, x2, y2 = map(
                            int,
                            caixa.xyxy[0]
                        )

                        # Converte as coordenadas da imagem
                        # girada para a imagem original.
                        novo_x1 = y1
                        novo_y1 = altura - x2
                        novo_x2 = y2
                        novo_y2 = altura - x1

                        melhor_caixa = (
                            novo_x1,
                            novo_y1,
                            novo_x2,
                            novo_y2
                        )


    # ======================================================
    # 5 - CONFIRMAÇÃO DA GARRAFA
    # ======================================================

    historico_garrafa.append(garrafa_detectada)

    quantidade_deteccoes = sum(historico_garrafa)

    garrafa_confirmada = (
        quantidade_deteccoes >= FRAMES_NECESSARIOS
    )


    # ======================================================
    # 6 - DESENHA A GARRAFA
    # ======================================================

    if (
        garrafa_detectada
        and melhor_caixa is not None
    ):

        desenhar_garrafa(
            frame,
            melhor_caixa,
            maior_confianca
        )


    # ======================================================
    # 7 - ALERTA DE LIXO
    # ======================================================

    if garrafa_confirmada:

        cv2.putText(
            frame,
            "LIXO DETECTADO",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        agora = time.time()

        if (
            agora - ultimo_alerta_lixo
            > INTERVALO_ALERTA
        ):

            print("======================================")
            print("ALERTA: LIXO DETECTADO")
            print("======================================")

            alerta_sonoro()

            ultimo_alerta_lixo = agora

    elif garrafa_detectada:

        cv2.putText(
            frame,
            "ANALISANDO GARRAFA...",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ======================================================
    # 8 - ALERTA DE MOVIMENTO DA ÁGUA
    # ======================================================
    # Mantido somente como lógica de alerta.
    # Os antigos textos/retângulos de ondulação foram removidos.

    if (
        deteccao_agua
        and confianca_agua >= 2
    ):

        agora = time.time()

        if (
            agora - ultimo_alerta_agua
            > INTERVALO_ALERTA
        ):

            print("======================================")
            print("ALERTA: MOVIMENTO FORTE NA ÁGUA")
            print("======================================")

            alerta_sonoro()

            ultimo_alerta_agua = agora


    # ======================================================
    # 9 - INFORMAÇÕES MÍNIMAS NA TELA
    # ======================================================

    contador = (
        f"Garrafas: "
        f"{quantidade_deteccoes}/{TOTAL_FRAMES}"
    )

    cv2.putText(
        frame,
        contador,
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================================
    # 10 - ROI
    # ======================================================

    cv2.rectangle(
        frame,
        (ROI_X, ROI_Y),
        (ROI_X + ROI_W, ROI_Y + ROI_H),
        (255, 0, 0),
        2
    )


    # ======================================================
    # 11 - EXIBIÇÃO PRINCIPAL
    # ======================================================

    cv2.imshow(
        "S.A.P.A. - Detector de Agua e Lixo",
        frame
    )


    # ======================================================
    # 12 - ATUALIZA FRAME
    # ======================================================

    prvs = atual


    # ======================================================
    # ESC PARA SAIR
    # ======================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================================
# FINALIZAÇÃO
# ==========================================================

cap.release()
cv2.destroyAllWindows()

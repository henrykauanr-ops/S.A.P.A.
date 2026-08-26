import cv2
import numpy as np
import time
import threading
from playsound import playsound
from collections import deque


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

# DroidCam
INDICE_CAMERA = 1


# Região onde está a água
# Ajuste estes valores se necessário
ROI_X = 100
ROI_Y = 80
ROI_W = 1080
ROI_H = 600


# ----------------------------------------------------------
# DETECÇÃO DA ÁGUA
# ----------------------------------------------------------

LIMIAR_MOVIMENTO = 1.0
AREA_MINIMA_AGUA = 5000


# ----------------------------------------------------------
# DETECÇÃO DE LIXO
# ----------------------------------------------------------

AREA_MINIMA_LIXO = 500
AREA_MAXIMA_LIXO = 25000

# Quantidade de frames necessários
# para confirmar que existe um objeto
FRAMES_LIXO = 5

# Histórico
historico_lixo = deque(maxlen=10)


# ----------------------------------------------------------
# ALERTAS
# ----------------------------------------------------------

ultimo_alerta_agua = 0
ultimo_alerta_lixo = 0

INTERVALO_ALERTA = 20


# ==========================================================
# ÁUDIO
# ==========================================================

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
# CÂMERA — DROIDCAM
# ==========================================================

cap = cv2.VideoCapture(
    INDICE_CAMERA,
    cv2.CAP_MSMF
)


if not cap.isOpened():

    print("ERRO: não foi possível abrir a DroidCam.")

    exit()


ret, frame1 = cap.read()


if not ret:

    print(
        "ERRO: DroidCam abriu, "
        "mas não enviou imagem."
    )

    cap.release()

    exit()


print("======================================")
print("DROIDCAM CONECTADA COM SUCESSO")
print("======================================")

print(
    "Resolucao:",
    frame1.shape[1],
    "x",
    frame1.shape[0]
)


# ==========================================================
# VERIFICAÇÃO DA ROI
# ==========================================================

altura, largura = frame1.shape[:2]


if ROI_X + ROI_W > largura:

    ROI_W = largura - ROI_X


if ROI_Y + ROI_H > altura:

    ROI_H = altura - ROI_Y


print(
    f"ROI: X={ROI_X}, Y={ROI_Y}, "
    f"W={ROI_W}, H={ROI_H}"
)


# ==========================================================
# PRIMEIRO FRAME
# ==========================================================

roi1 = frame1[
    ROI_Y:ROI_Y + ROI_H,
    ROI_X:ROI_X + ROI_W
]


prvs = cv2.cvtColor(
    roi1,
    cv2.COLOR_BGR2GRAY
)


historico_movimento = deque(maxlen=30)


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

while True:

    ret, frame2 = cap.read()


    if not ret:

        print("Erro ao receber imagem da DroidCam.")

        break


    # ======================================================
    # ROI
    # ======================================================

    roi2 = frame2[
        ROI_Y:ROI_Y + ROI_H,
        ROI_X:ROI_X + ROI_W
    ]


    # ======================================================
    # MOVIMENTO DA ÁGUA
    # ======================================================

    next_frame = cv2.cvtColor(
        roi2,
        cv2.COLOR_BGR2GRAY
    )


    next_frame = cv2.GaussianBlur(
        next_frame,
        (7, 7),
        0
    )


    # ======================================================
    # OPTICAL FLOW
    # ======================================================

    flow = cv2.calcOpticalFlowFarneback(

        prvs,

        next_frame,

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


    # ======================================================
    # ANÁLISE DO MOVIMENTO
    # ======================================================

    movimento_medio = np.mean(mag)

    historico_movimento.append(
        movimento_medio
    )


    media_historica = np.mean(
        historico_movimento
    )


    confianca_agua = 0


    if media_historica > LIMIAR_MOVIMENTO:

        confianca_agua += 1


    horizontal = np.mean(
        np.abs(flow[..., 0])
    )


    vertical = np.mean(
        np.abs(flow[..., 1])
    )


    if horizontal > vertical:

        confianca_agua += 1


    # ======================================================
    # MÁSCARA DE MOVIMENTO
    # ======================================================

    _, movimento = cv2.threshold(

        mag,

        LIMIAR_MOVIMENTO,

        255,

        cv2.THRESH_BINARY
    )


    movimento = movimento.astype(
        np.uint8
    )


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


    # ======================================================
    # CONTORNOS DA ÁGUA
    # ======================================================

    contours, _ = cv2.findContours(

        movimento,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    deteccao_agua = False


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < AREA_MINIMA_AGUA:

            continue


        deteccao_agua = True


        x, y, w, h = cv2.boundingRect(
            contour
        )


        cv2.rectangle(

            roi2,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )


    # ======================================================
    # DETECÇÃO DE POSSÍVEL LIXO
    # ======================================================

    hsv = cv2.cvtColor(

        roi2,

        cv2.COLOR_BGR2HSV
    )


    # ------------------------------------------------------
    # BLUR PARA REDUZIR RUÍDO
    # ------------------------------------------------------

    hsv_blur = cv2.GaussianBlur(

        hsv,

        (9, 9),

        0
    )


    # ======================================================
    # COMPARAÇÃO COM A REGIÃO AO REDOR
    # ======================================================

    # Média da imagem

    media_hsv = np.mean(

        hsv_blur.reshape(-1, 3),

        axis=0
    )


    H_media = int(
        media_hsv[0]
    )

    S_media = int(
        media_hsv[1]
    )

    V_media = int(
        media_hsv[2]
    )


    # ======================================================
    # DIFERENÇA DE COR
    # ======================================================

    diferenca_h = cv2.absdiff(

        hsv_blur[:, :, 0],

        np.full(

            hsv_blur[:, :, 0].shape,

            H_media,

            dtype=np.uint8
        )
    )


    diferenca_s = cv2.absdiff(

        hsv_blur[:, :, 1],

        np.full(

            hsv_blur[:, :, 1].shape,

            S_media,

            dtype=np.uint8
        )
    )


    diferenca_v = cv2.absdiff(

        hsv_blur[:, :, 2],

        np.full(

            hsv_blur[:, :, 2].shape,

            V_media,

            dtype=np.uint8
        )
    )


    # ======================================================
    # MÁSCARA DE OBJETOS DIFERENTES
    # ======================================================

    mascara_lixo = (

        (diferenca_h > 18) |

        (diferenca_s > 45) |

        (diferenca_v > 50)
    )


    mascara_lixo = (

        mascara_lixo.astype(
            np.uint8
        )

    ) * 255


    # ======================================================
    # LIMPEZA
    # ======================================================

    kernel_lixo = np.ones(

        (7, 7),

        np.uint8
    )


    mascara_lixo = cv2.morphologyEx(

        mascara_lixo,

        cv2.MORPH_OPEN,

        kernel_lixo
    )


    mascara_lixo = cv2.morphologyEx(

        mascara_lixo,

        cv2.MORPH_CLOSE,

        kernel_lixo
    )


    # ======================================================
    # CONTORNOS DOS OBJETOS
    # ======================================================

    contours_lixo, _ = cv2.findContours(

        mascara_lixo,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    possivel_lixo = False


    # ======================================================
    # ANALISA CADA OBJETO
    # ======================================================

    for contour in contours_lixo:

        area = cv2.contourArea(
            contour
        )


        # Ignora objetos pequenos

        if area < AREA_MINIMA_LIXO:

            continue


        # Ignora regiões muito grandes

        if area > AREA_MAXIMA_LIXO:

            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Dimensão mínima

        if w < 20 or h < 20:

            continue


        # ==================================================
        # PROPORÇÃO
        # ==================================================

        proporcao = w / float(h)


        if proporcao < 0.2:

            continue


        if proporcao > 5:

            continue


        # ==================================================
        # OBJETO ACEITO
        # ==================================================

        possivel_lixo = True


        cv2.rectangle(

            roi2,

            (x, y),

            (x + w, y + h),

            (0, 255, 255),

            3
        )


        cv2.putText(

            roi2,

            "POSSIVEL LIXO",

            (x, max(y - 10, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


        # Apenas um objeto por frame

        break


    # ======================================================
    # HISTÓRICO DO LIXO
    # ======================================================

    if possivel_lixo:

        historico_lixo.append(1)

    else:

        historico_lixo.append(0)


    quantidade_lixo = sum(
        historico_lixo
    )


    lixo_confirmado = (

        quantidade_lixo >= FRAMES_LIXO
    )


    # ======================================================
    # ALERTA DE LIXO
    # ======================================================

    if lixo_confirmado:

        cv2.putText(

            frame2,

            "LIXO DETECTADO",

            (50, 100),

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

            print(
                "ALERTA: LIXO DETECTADO"
            )


            alerta_sonoro()


            ultimo_alerta_lixo = agora


    # ======================================================
    # ALERTA DE ÁGUA
    # ======================================================

    if (

        deteccao_agua

        and

        confianca_agua >= 2

    ):

        cv2.putText(

            frame2,

            "ONDULACAO DETECTADA",

            (50, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            3
        )


        agora = time.time()


        if (

            agora - ultimo_alerta_agua

            > INTERVALO_ALERTA

        ):

            print(
                "ALERTA REAL DETECTADO"
            )


            alerta_sonoro()


            ultimo_alerta_agua = agora


    # ======================================================
    # DESENHA ROI
    # ======================================================

    cv2.rectangle(

        frame2,

        (ROI_X, ROI_Y),

        (
            ROI_X + ROI_W,
            ROI_Y + ROI_H
        ),

        (255, 0, 0),

        2
    )


    # ======================================================
    # INFORMAÇÕES
    # ======================================================

    cv2.putText(

        frame2,

        f"Movimento: {media_historica:.2f}",

        (50, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    if lixo_confirmado:

        cv2.putText(

            frame2,

            "STATUS: LIXO DETECTADO",

            (50, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )

    else:

        cv2.putText(

            frame2,

            "STATUS: AGUA NORMAL",

            (50, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


    # ======================================================
    # EXIBIÇÃO
    # ======================================================

    cv2.imshow(
        "S.A.P.A. - DroidCam",
        frame2
    )


    cv2.imshow(
        "Movimento da Agua",
        movimento
    )


    cv2.imshow(
        "Deteccao de Possiveis Residuos",
        mascara_lixo
    )


    # ======================================================
    # ATUALIZA FRAME
    # ======================================================

    prvs = next_frame


    # ESC
    if cv2.waitKey(1) & 0xFF == 27:

        break


# ==========================================================
# FINALIZAÇÃO
# ==========================================================

cap.release()

cv2.destroyAllWindows()
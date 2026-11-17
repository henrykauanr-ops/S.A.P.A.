import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from playsound import playsound
from ultralytics import YOLO


# ==========================================================
# S.A.P.A. - SISTEMA DE ALERTA E PREVISÃO DE ALAGAMENTO (v1.6)
# Detecção de movimento da água + detecção de garrafões/lixo
# com suporte a modelo customizado (best.pt) e inferência na ROI
# ==========================================================


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

DIRETORIO_ATUAL = Path(__file__).resolve().parent

INDICE_CAMERA = 0
FONTE_VIDEO = None  # Ex.: "videos/teste_rio.mp4" ou passe via --video

MODELO_CUSTOM = DIRETORIO_ATUAL / "best.pt"
MODELO_FALLBACK = "yolo11n.pt"

CLASSES_CUSTOM = {"garrafao", "garrafa_pet"}
CLASSES_FALLBACK = {"bottle", "vase", "cup"}

# Confiança mínima ajustada para detectar objetos parcialmente submersos
CONFIANCA_MINIMA = 0.55

# Confirmação temporal:
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

AUDIO_ALERTA = str(DIRETORIO_ATUAL / "ssstik.io_1778940651943.mp3")

audio_tocando = False


def tocar_audio():
    """Executa o áudio de alerta em segundo plano."""
    global audio_tocando

    try:
        audio_tocando = True
        playsound(AUDIO_ALERTA)

    except Exception as erro:
        print(f"Erro ao tocar áudio: {erro}")

    finally:
        audio_tocando = False


def alerta_sonoro():
    """Dispara a thread de alerta sonoro sem bloquear o vídeo."""
    if not audio_tocando:
        thread_audio = threading.Thread(
            target=tocar_audio,
            daemon=True,
        )
        thread_audio.start()


# ==========================================================
# MODELO YOLO (CUSTOMIZADO OU FALLBACK)
# ==========================================================

def carregar_modelo_yolo():
    """Carrega best.pt se disponível; caso contrário usa yolo11n.pt em fallback."""
    if MODELO_CUSTOM.exists():
        print(f"Modelo customizado encontrado: {MODELO_CUSTOM.name}")
        return YOLO(str(MODELO_CUSTOM)), CLASSES_CUSTOM, True

    print(
        f"Aviso: '{MODELO_CUSTOM.name}' não encontrado. "
        f"Usando fallback '{MODELO_FALLBACK}' (classes: {sorted(CLASSES_FALLBACK)})."
    )
    return YOLO(MODELO_FALLBACK), CLASSES_FALLBACK, False


# ==========================================================
# DESENHA DETECÇÃO DA GARRAFA / RESÍDUO
# ==========================================================

def desenhar_garrafa(imagem, caixa, confianca, rotulo="GARRAFA"):
    """Desenha a bounding box e o rótulo de confiança no frame principal."""
    x1, y1, x2, y2 = map(int, caixa)

    cv2.rectangle(
        imagem,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    texto = f"{rotulo.upper()} {confianca * 100:.1f}%"

    cv2.putText(
        imagem,
        texto,
        (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def parse_args():
    """Lê argumentos opcionais para alternar entre webcam e arquivo .mp4."""
    parser = argparse.ArgumentParser(
        description="S.A.P.A. v1.6 - Monitoramento de água e detecção de resíduos."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=FONTE_VIDEO,
        help="Caminho para arquivo de vídeo (.mp4) para testar sem câmera.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=INDICE_CAMERA,
        help="Índice da câmera/webcam (padrão: 0).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reinicia o vídeo automaticamente ao chegar no final.",
    )
    return parser.parse_args()


# ==========================================================
# INICIALIZAÇÃO
# ==========================================================

def main():
    global ROI_W, ROI_H, ultimo_alerta_agua, ultimo_alerta_lixo

    args = parse_args()
    modelo, classes_alvo, usando_custom = carregar_modelo_yolo()

    fonte = args.video if args.video else args.camera
    usando_video = args.video is not None

    cap = cv2.VideoCapture(fonte)

    if not cap.isOpened():
        print(f"ERRO: não foi possível abrir a fonte de vídeo ({fonte}).")
        return

    ret, primeiro_frame = cap.read()

    if not ret:
        print("ERRO: a fonte abriu, mas não enviou imagem.")
        cap.release()
        return

    altura, largura = primeiro_frame.shape[:2]

    print("======================================")
    print("S.A.P.A. v1.6 INICIADO COM SUCESSO")
    print("======================================")
    print(f"Fonte:     {'Vídeo (' + str(fonte) + ')' if usando_video else f'Câmera {fonte}'}")
    print(f"Modo YOLO: {'Customizado (best.pt)' if usando_custom else 'Fallback (yolo11n.pt)'}")
    print(f"Resolução: {largura} x {altura}")

    # ======================================================
    # AJUSTE DA ROI
    # ======================================================

    if ROI_X + ROI_W > largura:
        ROI_W = largura - ROI_X

    if ROI_Y + ROI_H > altura:
        ROI_H = altura - ROI_Y

    if ROI_W <= 0 or ROI_H <= 0:
        print("ERRO: ROI inválida para a resolução da câmera/vídeo.")
        cap.release()
        return

    print(f"ROI: X={ROI_X}, Y={ROI_Y}, W={ROI_W}, H={ROI_H}")

    # ======================================================
    # PRIMEIRO FRAME PARA O MOVIMENTO DA ÁGUA
    # ======================================================

    roi1 = primeiro_frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    prvs = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    prvs = cv2.GaussianBlur(prvs, (7, 7), 0)

    historico_movimento = deque(maxlen=30)

    # ======================================================
    # LOOP PRINCIPAL
    # ======================================================

    while True:
        ret, frame = cap.read()

        if not ret:
            if usando_video and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Fim da transmissão/vídeo.")
            break

        # ==================================================
        # 1 - DETECÇÃO DE MOVIMENTO DA ÁGUA (OPTICAL FLOW)
        # ==================================================

        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

        atual = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        atual = cv2.GaussianBlur(atual, (7, 7), 0)

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
            0,
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        movimento_medio = float(np.mean(mag))
        historico_movimento.append(movimento_medio)
        media_historica = float(np.mean(historico_movimento))

        horizontal = float(np.mean(np.abs(flow[..., 0])))
        vertical = float(np.mean(np.abs(flow[..., 1])))

        confianca_agua = 0
        if media_historica > LIMIAR_MOVIMENTO:
            confianca_agua += 1
        if horizontal > vertical:
            confianca_agua += 1

        # ==================================================
        # 2 - MÁSCARA DO MOVIMENTO DA ÁGUA
        # ==================================================

        _, movimento = cv2.threshold(
            mag,
            LIMIAR_MOVIMENTO,
            255,
            cv2.THRESH_BINARY,
        )
        movimento = movimento.astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        movimento = cv2.morphologyEx(movimento, cv2.MORPH_OPEN, kernel)
        movimento = cv2.dilate(movimento, kernel, iterations=2)

        contours, _ = cv2.findContours(
            movimento,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        deteccao_agua = False
        for contour in contours:
            if cv2.contourArea(contour) >= AREA_MINIMA_AGUA:
                deteccao_agua = True
                break

        # ==================================================
        # 3 - DETECÇÃO DE LIXO COM YOLO APENAS NA ROI
        # ==================================================

        garrafa_detectada = False
        deteccoes_frame = []

        resultados = modelo(
            roi,
            conf=CONFIANCA_MINIMA,
            verbose=False,
        )

        for resultado in resultados:
            for caixa in resultado.boxes:
                classe = int(caixa.cls[0])
                confianca = float(caixa.conf[0])
                nome = modelo.names[classe]

                if nome in classes_alvo and confianca >= CONFIANCA_MINIMA:
                    garrafa_detectada = True

                    # Converte coordenadas da ROI para o frame principal
                    x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                    caixa_frame = (
                        x1 + ROI_X,
                        y1 + ROI_Y,
                        x2 + ROI_X,
                        y2 + ROI_Y,
                    )
                    deteccoes_frame.append((caixa_frame, confianca, nome))

        # ==================================================
        # 4 - CONFIRMAÇÃO TEMPORAL
        # ==================================================

        historico_garrafa.append(garrafa_detectada)
        quantidade_deteccoes = sum(historico_garrafa)
        garrafa_confirmada = quantidade_deteccoes >= FRAMES_NECESSARIOS

        # ==================================================
        # 5 - DESENHA AS DETECÇÕES NO FRAME PRINCIPAL
        # ==================================================

        for caixa_frame, confianca, nome in deteccoes_frame:
            desenhar_garrafa(frame, caixa_frame, confianca, rotulo=nome)

        # ==================================================
        # 6 - ALERTA DE LIXO
        # ==================================================

        if garrafa_confirmada:
            cv2.putText(
                frame,
                "LIXO DETECTADO",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )

            agora = time.time()
            if agora - ultimo_alerta_lixo > INTERVALO_ALERTA:
                print("======================================")
                print("ALERTA: LIXO DETECTADO")
                print("======================================")
                alerta_sonoro()
                ultimo_alerta_lixo = agora

        elif garrafa_detectada:
            cv2.putText(
                frame,
                "ANALISANDO RESIDUO...",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # ==================================================
        # 7 - ALERTA DE MOVIMENTO DA ÁGUA
        # ==================================================

        if deteccao_agua and confianca_agua >= 2:
            agora = time.time()
            if agora - ultimo_alerta_agua > INTERVALO_ALERTA:
                print("======================================")
                print("ALERTA: MOVIMENTO FORTE NA ÁGUA")
                print("======================================")
                alerta_sonoro()
                ultimo_alerta_agua = agora

        # ==================================================
        # 8 - INFORMAÇÕES MÍNIMAS NA TELA
        # ==================================================

        contador = f"Residuos: {quantidade_deteccoes}/{TOTAL_FRAMES}"
        cv2.putText(
            frame,
            contador,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # ==================================================
        # 9 - ROI E EXIBIÇÃO PRINCIPAL
        # ==================================================

        cv2.rectangle(
            frame,
            (ROI_X, ROI_Y),
            (ROI_X + ROI_W, ROI_Y + ROI_H),
            (255, 0, 0),
            2,
        )

        cv2.imshow("S.A.P.A. - Detector de Agua e Lixo", frame)

        # ==================================================
        # 10 - ATUALIZA FRAME E VERIFICA ESC
        # ==================================================

        prvs = atual

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ======================================================
    # FINALIZAÇÃO
    # ======================================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

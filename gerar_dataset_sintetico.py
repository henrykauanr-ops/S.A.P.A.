"""Gerador de dataset sintético offline (Data Augmentation) para o S.A.P.A.

Multiplica imagens reais anotadas no formato YOLO gerando variações:
- Rotações de 0° a 360° em passos configuráveis;
- Espelhamentos horizontal e vertical;
- Variações de brilho, contraste e reflexo aquático;
- Oclusão parcial na base dos objetos (simulando garrafões/garrafas meio submersos).

Recalcula automaticamente todas as bounding boxes no formato YOLO.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIRETORIO_RAIZ = Path(__file__).resolve().parent

# Tipo: (class_id, cx, cy, w, h) normalizados entre 0 e 1
YoloBox = Tuple[int, float, float, float, float]


def ler_labels_yolo(caminho_label: Path) -> List[YoloBox]:
    """Lê um arquivo de anotação YOLO (.txt) e retorna lista de bounding boxes."""
    if not caminho_label.exists():
        return []

    caixas: List[YoloBox] = []
    for linha in caminho_label.read_text(encoding="utf-8").splitlines():
        partes = linha.strip().split()
        if len(partes) != 5:
            continue
        cls_id = int(float(partes[0]))
        cx, cy, w, h = map(float, partes[1:])
        caixas.append((cls_id, cx, cy, w, h))
    return caixas


def salvar_par(
    imagem: np.ndarray,
    caixas: List[YoloBox],
    pasta_imagens: Path,
    pasta_labels: Path,
    nome_base: str,
) -> None:
    """Salva uma imagem aumentada e seu respectivo arquivo de labels YOLO."""
    caminho_img = pasta_imagens / f"{nome_base}.jpg"
    caminho_lbl = pasta_labels / f"{nome_base}.txt"

    cv2.imwrite(str(caminho_img), imagem)

    linhas = [
        f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls_id, cx, cy, w, h in caixas
    ]
    caminho_lbl.write_text("\n".join(linhas), encoding="utf-8")


def rotacionar_imagem_e_caixas(
    imagem: np.ndarray,
    caixas: List[YoloBox],
    angulo: float,
) -> Tuple[np.ndarray, List[YoloBox]]:
    """Rotaciona imagem expandindo o canvas para evitar cortes e recalcula as caixas YOLO."""
    altura, largura = imagem.shape[:2]
    centro = (largura / 2.0, altura / 2.0)

    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos_a = abs(matriz[0, 0])
    sin_a = abs(matriz[0, 1])

    nova_largura = int((altura * sin_a) + (largura * cos_a))
    nova_altura = int((altura * cos_a) + (largura * sin_a))

    matriz[0, 2] += (nova_largura / 2.0) - centro[0]
    matriz[1, 2] += (nova_altura / 2.0) - centro[1]

    # Usa a cor mediana das bordas para evitar objetos "fantasma" por reflexão
    bordas = np.concatenate([
        imagem[0, :],
        imagem[-1, :],
        imagem[:, 0],
        imagem[:, -1],
    ], axis=0)
    cor_borda = tuple(int(c) for c in np.median(bordas, axis=0))

    img_rot = cv2.warpAffine(
        imagem,
        matriz,
        (nova_largura, nova_altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=cor_borda,
    )

    novas_caixas: List[YoloBox] = []
    for cls_id, cx, cy, w, h in caixas:
        x_c = cx * largura
        y_c = cy * altura
        bw = w * largura
        bh = h * altura

        xmin, xmax = x_c - bw / 2.0, x_c + bw / 2.0
        ymin, ymax = y_c - bh / 2.0, y_c + bh / 2.0

        cantos = np.array([
            [xmin, ymin, 1.0],
            [xmax, ymin, 1.0],
            [xmax, ymax, 1.0],
            [xmin, ymax, 1.0],
        ])
        cantos_rot = cantos @ matriz.T

        nx1 = max(0.0, float(np.min(cantos_rot[:, 0])))
        ny1 = max(0.0, float(np.min(cantos_rot[:, 1])))
        nx2 = min(float(nova_largura), float(np.max(cantos_rot[:, 0])))
        ny2 = min(float(nova_altura), float(np.max(cantos_rot[:, 1])))

        nw = nx2 - nx1
        nh = ny2 - ny1
        if nw < 6.0 or nh < 6.0:
            continue

        ncx = (nx1 + nx2) / (2.0 * nova_largura)
        ncy = (ny1 + ny2) / (2.0 * nova_altura)
        novas_caixas.append((cls_id, ncx, ncy, nw / nova_largura, nh / nova_altura))

    return img_rot, novas_caixas


def espelhar_imagem_e_caixas(
    imagem: np.ndarray,
    caixas: List[YoloBox],
    horizontal: bool = True,
) -> Tuple[np.ndarray, List[YoloBox]]:
    """Espelha imagem (horizontal ou vertical) e ajusta as coordenadas YOLO."""
    if horizontal:
        img_flip = cv2.flip(imagem, 1)
        novas = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in caixas]
    else:
        img_flip = cv2.flip(imagem, 0)
        novas = [(c, cx, 1.0 - cy, w, h) for c, cx, cy, w, h in caixas]
    return img_flip, novas


def aplicar_variacao_iluminacao(
    imagem: np.ndarray,
    fator_brilho: float,
    adicionar_reflexo: bool = False,
) -> np.ndarray:
    """Altera brilho/contraste e opcionalmente adiciona reflexo simulando água."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * fator_brilho, 0, 255)
    resultado = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if adicionar_reflexo:
        altura, largura = resultado.shape[:2]
        camada = resultado.copy()
        cx = random.randint(int(largura * 0.2), int(largura * 0.8))
        cy = random.randint(int(altura * 0.2), int(altura * 0.8))
        eixo_x = random.randint(int(largura * 0.08), int(largura * 0.22))
        eixo_y = random.randint(6, max(10, int(altura * 0.05)))
        cv2.ellipse(
            camada,
            (cx, cy),
            (eixo_x, eixo_y),
            random.uniform(-15, 15),
            0,
            360,
            (245, 250, 255),
            -1,
        )
        resultado = cv2.addWeighted(camada, 0.28, resultado, 0.72, 0)

    return resultado


def aplicar_submersao_parcial(
    imagem: np.ndarray,
    caixas: List[YoloBox],
    fracao_submersa: float = 0.35,
) -> np.ndarray:
    """Simula oclusão parcial da base dos objetos como se estivessem submersos."""
    altura, largura = imagem.shape[:2]
    resultado = imagem.copy()

    for _, cx, cy, w, h in caixas:
        xmin = max(0, int((cx - w / 2.0) * largura))
        xmax = min(largura, int((cx + w / 2.0) * largura))
        ymin = max(0, int((cy - h / 2.0) * altura))
        ymax = min(altura, int((cy + h / 2.0) * altura))

        altura_box = ymax - ymin
        if altura_box < 12 or (xmax - xmin) < 12:
            continue

        y_corte = ymax - int(altura_box * fracao_submersa)
        regiao = resultado[y_corte:ymax, xmin:xmax]
        if regiao.size == 0:
            continue

        # Mistura a parte inferior com blur + tonalidade aquática
        borrada = cv2.GaussianBlur(regiao, (15, 5), 0)
        cor_agua = np.full_like(borrada, (110, 120, 85))
        efeito_agua = cv2.addWeighted(borrada, 0.45, cor_agua, 0.55, 0)
        resultado[y_corte:ymax, xmin:xmax] = efeito_agua

    return resultado


def parse_args() -> argparse.Namespace:
    """Processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera variações sintéticas (rotação, brilho, submersão) para dataset YOLO."
    )
    parser.add_argument(
        "--input-images",
        type=Path,
        default=DIRETORIO_RAIZ / "dataset" / "raw" / "images",
        help="Pasta com as imagens originais.",
    )
    parser.add_argument(
        "--input-labels",
        type=Path,
        default=DIRETORIO_RAIZ / "dataset" / "raw" / "labels",
        help="Pasta com os arquivos .txt originais no formato YOLO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIRETORIO_RAIZ / "dataset",
        help="Diretório base de saída (criará images/train e labels/train).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=45,
        help="Passo de rotação em graus entre 0 e 360 (padrão: 45).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o pipeline de data augmentation offline."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.input_images.exists():
        print(f"ERRO: Pasta de imagens de entrada não encontrada: {args.input_images}")
        print("Crie a pasta e coloque as fotos originais + labels YOLO antes de rodar.")
        return

    out_images = args.output_dir / "images" / "train"
    out_labels = args.output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    arquivos_img = sorted(
        p for p in args.input_images.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM
    )

    if not arquivos_img:
        print(f"Nenhuma imagem encontrada em: {args.input_images}")
        return

    total_geradas = 0
    angulos = list(range(0, 360, max(1, args.step)))

    for caminho_img in arquivos_img:
        imagem = cv2.imread(str(caminho_img))
        if imagem is None:
            print(f"Aviso: não foi possível ler {caminho_img.name}, pulando.")
            continue

        caixas = ler_labels_yolo(args.input_labels / f"{caminho_img.stem}.txt")
        if not caixas:
            print(f"Aviso: label ausente ou vazia para {caminho_img.name}, pulando.")
            continue

        stem = caminho_img.stem

        # 1. Rotações 0°..360°
        for angulo in angulos:
            img_rot, caixas_rot = rotacionar_imagem_e_caixas(imagem, caixas, float(angulo))
            if not caixas_rot:
                continue

            salvar_par(img_rot, caixas_rot, out_images, out_labels, f"{stem}_rot{angulo}")
            total_geradas += 1

            # Variação de brilho/reflexo em ângulos alternados
            if angulo % (args.step * 2) == 0:
                fator = random.uniform(0.65, 1.35)
                img_luz = aplicar_variacao_iluminacao(img_rot, fator, adicionar_reflexo=True)
                salvar_par(img_luz, caixas_rot, out_images, out_labels, f"{stem}_rot{angulo}_luz")
                total_geradas += 1

        # 2. Espelhamento horizontal e vertical
        img_h, caixas_h = espelhar_imagem_e_caixas(imagem, caixas, horizontal=True)
        salvar_par(img_h, caixas_h, out_images, out_labels, f"{stem}_fliph")
        total_geradas += 1

        img_v, caixas_v = espelhar_imagem_e_caixas(imagem, caixas, horizontal=False)
        salvar_par(img_v, caixas_v, out_images, out_labels, f"{stem}_flipv")
        total_geradas += 1

        # 3. Simulação de objeto parcialmente submerso
        img_sub = aplicar_submersao_parcial(imagem, caixas, fracao_submersa=random.uniform(0.28, 0.42))
        salvar_par(img_sub, caixas, out_images, out_labels, f"{stem}_submerso")
        total_geradas += 1

    print("======================================")
    print("DATA AUGMENTATION CONCLUÍDO")
    print("======================================")
    print(f"Imagens originais processadas: {len(arquivos_img)}")
    print(f"Total de amostras geradas:     {total_geradas}")
    print(f"Saída salva em:                {out_images}")


if __name__ == "__main__":
    main()

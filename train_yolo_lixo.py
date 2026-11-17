"""Pipeline de fine-tuning do YOLO11n para detecção de resíduos na água (S.A.P.A.).

Treina o modelo para reconhecer garrafões (20L) e garrafas PET em qualquer
ângulo, orientação ou condição de iluminação aquática, eliminando a necessidade
de múltiplas passagens de inferência com rotação manual em produção.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIRETORIO_RAIZ = Path(__file__).resolve().parent


def contar_imagens(diretorio: Path) -> int:
    """Conta quantas imagens válidas existem em um diretório."""
    if not diretorio.exists():
        return 0
    return sum(
        1 for arquivo in diretorio.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM
    )


def validar_dataset(caminho_data_yaml: Path) -> bool:
    """Verifica se o data.yaml e as pastas de treino/validação contêm imagens."""
    if not caminho_data_yaml.exists():
        print(f"ERRO: Arquivo de configuração não encontrado: {caminho_data_yaml}")
        return False

    pasta_dataset = caminho_data_yaml.parent
    pasta_train = pasta_dataset / "images" / "train"
    pasta_val = pasta_dataset / "images" / "val"

    qtd_train = contar_imagens(pasta_train)
    qtd_val = contar_imagens(pasta_val)

    if qtd_train == 0 or qtd_val == 0:
        print("======================================")
        print("DATASET INCOMPLETO PARA TREINAMENTO")
        print("======================================")
        print(f"Imagens em train ({pasta_train}): {qtd_train}")
        print(f"Imagens em val   ({pasta_val}): {qtd_val}")
        print(
            "\nAdicione imagens e arquivos .txt no formato YOLO antes de treinar.\n"
            "Dica: você pode multiplicar poucas fotos reais usando:\n"
            "  python gerar_dataset_sintetico.py"
        )
        return False

    print(f"Dataset validado: {qtd_train} imagens em train | {qtd_val} em val.")
    return True


def parse_args() -> argparse.Namespace:
    """Processa argumentos de linha de comando para o treinamento."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning YOLO11n para detecção de garrafões e garrafas PET (S.A.P.A.)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DIRETORIO_RAIZ / "dataset" / "data.yaml",
        help="Caminho para o arquivo data.yaml do dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Modelo base para transferência de aprendizado (padrão: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas de treinamento (padrão: 100).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho da imagem de entrada para o YOLO (padrão: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Tamanho do batch (reduza para 8 ou 4 se faltar VRAM; padrão: 16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo ('0' para GPU, 'cpu' para CPU, vazio para automático).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sapa_lixo",
        help="Nome do experimento dentro de runs/detect/ (padrão: sapa_lixo).",
    )
    parser.add_argument(
        "--copiar-best",
        action="store_true",
        help="Copia automaticamente o best.pt gerado para a raiz do projeto.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o pipeline completo de fine-tuning do YOLO11n."""
    args = parse_args()
    caminho_data = args.data.resolve()

    if not validar_dataset(caminho_data):
        return

    print("======================================")
    print("INICIANDO TREINAMENTO S.A.P.A. (YOLO11n)")
    print("======================================")

    modelo = YOLO(args.model)

    parametros_treino = {
        "data": str(caminho_data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(DIRETORIO_RAIZ / "runs" / "detect"),
        "name": args.name,
        "patience": 25,
        # Augmentations para invariância de rotação e iluminação aquática
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "scale": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.4,
    }

    if args.device:
        parametros_treino["device"] = args.device

    resultados = modelo.train(**parametros_treino)

    caminho_best = Path(resultados.save_dir) / "weights" / "best.pt"
    print("======================================")
    print("TREINAMENTO CONCLUÍDO")
    print("======================================")
    print(f"Melhor modelo salvo em: {caminho_best}")

    destino_raiz = DIRETORIO_RAIZ / "best.pt"
    if args.copiar_best and caminho_best.exists():
        shutil.copy2(caminho_best, destino_raiz)
        print(f"Modelo copiado para a raiz: {destino_raiz}")
    elif caminho_best.exists():
        print(
            "Para ativar este modelo no SAPA_protótipo_1.6.py, copie o arquivo para:\n"
            f"  {destino_raiz}"
        )


if __name__ == "__main__":
    main()

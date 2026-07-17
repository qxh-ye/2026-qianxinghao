import os
from pathlib import Path

from src.preprocess.enhancement import process_image


def main():
    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 数据路径
    raw_dir = BASE_DIR / "data" / "raw"
    save_dir = BASE_DIR / "data" / "processed"

    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(raw_dir):
        if img_name.endswith((".jpg", ".png")):
            img_path = os.path.join(raw_dir, img_name)

            process_image(img_path, save_dir)

if __name__ == "__main__":
    main()

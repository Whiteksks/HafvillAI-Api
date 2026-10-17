"""python -m training.prepare --config training/configs/consciousness.json"""

import argparse
import json

from .config import load_config
from .data import prepare


def main():
    parser = argparse.ArgumentParser(description="Подготовка SFT без изменения исходных датасетов")
    parser.add_argument("--config", required=True)
    parser.add_argument("--offline", action="store_true", help="Использовать только локальный кеш токенизатора")
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config), offline=args.offline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

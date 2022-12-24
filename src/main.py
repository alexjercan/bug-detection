import logging
import os

import tarfile
import wget

INPUT_PATH = "input"
ROOT_PATH = os.path.join(INPUT_PATH, "Project_CodeNet")
DATA_PATH = os.path.join(ROOT_PATH, "data")


def download_codenet(force: bool = False) -> None:
    data_url = "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0"
    tar_name = "Project_CodeNet.tar.gz"
    tar_path = os.path.join(INPUT_PATH, tar_name)

    if os.path.exists(ROOT_PATH) and not force:
        logging.info(f"dataset root dir found at {ROOT_PATH}. skiping...")
        return

    if not os.path.exists(tar_path) or force:
        logging.debug(f"download dataset from {data_url}/{tar_name}")
        wget.download(f"{data_url}/{tar_name}", out=tar_path)

    with tarfile.open(tar_path) as tar:
        logging.debug(f"extract codenet to {INPUT_PATH}")
        tar.extractall(path=INPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "codenet.log")),
        ],
        level=logging.DEBUG,
        format="%(levelname)s: %(asctime)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    download_codenet()

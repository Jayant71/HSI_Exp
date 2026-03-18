from ast import parse
import os
import sys
import urllib.request
import ssl
import argparse
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.config as config


os.makedirs(config.DATA_DIR, exist_ok=True)

def dataloader(dataset: str = "ip"):
    if dataset == "ip":
        HSI_FILE = os.path.join(config.DATA_DIR, "Indian_pines_corrected.mat")
        GT_FILE = os.path.join(config.DATA_DIR, "Indian_pines_gt.mat")
    elif dataset == "pu":
        HSI_FILE = os.path.join(config.DATA_DIR, "PaviaU.mat")
        GT_FILE = os.path.join(config.DATA_DIR, "PaviaU_gt.mat")
    elif dataset == "sa":
        HSI_FILE = os.path.join(config.DATA_DIR, "Salinas_corrected.mat")
        GT_FILE = os.path.join(config.DATA_DIR, "Salinas_gt.mat")

    hsi_mat = loadmat(HSI_FILE)
    gt_mat  = loadmat(GT_FILE)

    cube = hsi_mat[list(hsi_mat.keys())[-1]]  
    gt   = gt_mat[list(gt_mat.keys())[-1]]   


    print(f"Cube shape : {cube.shape}  (rows × cols × bands)")
    print(f"GT shape   : {gt.shape}")
    print(f"Classes    : {np.unique(gt)}")  

    return cube, gt




def download_file(url: str, dest: str) -> None:
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)} ...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        pbar = None
        def reporthook(blocknum, blocksize, totalsize):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=totalsize, unit='B', unit_scale=True, desc=os.path.basename(dest))
            downloaded = blocknum * blocksize
            pbar.update(downloaded - pbar.n)
        
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        if pbar:
            pbar.close()
        print(f"  Saved → {dest}")
    else:
        print(f"  {os.path.basename(dest)} already present, skipping download.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the dataset and ground truth.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if not already present.")
    parser.add_argument("--dataset", choices=["ip", "pu", "sa" ], default="ip", help="Specify which dataset to download (ip=Indian Pines, pu=PaviaU, sa=Salinas).")
    args = parser.parse_args()

    if args.download:
        if args.dataset == "ip":
            download_file(config.IP_URL, os.path.join(config.DATA_DIR, "Indian_pines_corrected.mat"))
            download_file(config.IP_GT_URL, os.path.join(config.DATA_DIR, "Indian_pines_gt.mat"))
        elif args.dataset == "pu":
            download_file(config.PU_URL, os.path.join(config.DATA_DIR, "PaviaU.mat"))
            download_file(config.PU_GT_URL, os.path.join(config.DATA_DIR, "PaviaU_gt.mat"))
        elif args.dataset == "sa":
            download_file(config.SA_URL, os.path.join(config.DATA_DIR, "Salinas_corrected.mat"))
            download_file(config.SA_GT_URL, os.path.join(config.DATA_DIR, "Salinas_gt.mat"))
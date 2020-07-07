"""Extract 100k img from .mdb into byte img in folder"""
import io
import os
import argparse

import lmdb
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

def set_args():
    parser = argparse.ArgumentParser("Export from .mdb file to flat folder")
    parser.add_argument("--db_path", type=str, default="../data_unzip/bedroom_train_lmdb/",
                        help="database folder")
    parser.add_argument("--df_path", type=str, default="lsun_100k.csv",
                        help="100k dataframe list path")
    parser.add_argument("--out_dir", type=str, default="../data_unzip/bedroom_train_lmdb/lsun_bed100k")
    
    args = parser.parse_args()

    return args

def export_images(df, db_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    name_list = list(df.iloc[:,0])
    for name in tqdm(name_list, total=len(name_list)):
        key = name.split(".")[0]
        byte_key = bytes(key, 'utf-8')
        with env.begin(write=False) as txn:
            byte_img = txn.get(byte_key)
            out_img_path = os.path.join(out_dir, key + '.webp')
            with open(out_img_path, 'wb') as f:
                f.write(byte_img)
    return out_img_path 





def main():
    args = set_args()
    df = pd.read_csv(args.df_path, index_col=0)
    last_img_path = export_images(df, args.db_path, args.out_dir)

    # test
    print("Testing results")
    img = Image.open(last_img_path)
    img = np.array(img)
    print("Last img shape: {} ---- OK!".format(img.shape))



if __name__ == "__main__":
    """Command
    $ python to_100k.py --db_path xxxx/bedroom_train_lmdb --df_path lsun_100k.csv --out_dir /path/to/save/dir
    """
    main()

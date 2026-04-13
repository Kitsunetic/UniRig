import argparse
import os

import yaml
from box import Box
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config = Box(yaml.safe_load(open(args.config, "r")))

    dataset = config.output_dataset_dir

    paths = []
    for root, dirs, files in tqdm(os.walk(dataset)):
        for file in files:
            if file == "raw_data.npz":
                paths.append(os.path.relpath(root, dataset))
    tot = len(paths)
    os.makedirs(dataset, exist_ok=True)
    f = open(os.path.join(dataset, f"inference_datalist.txt"), "w")
    f.writelines("\n".join(paths))
    f.close()

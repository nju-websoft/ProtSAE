import argparse
from pathlib import Path

import numcodecs
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from protsae.data import ONTOLOGIES, collate_protein_batch, load_deepgo_sequence_dataset
from protsae.esm import (
    ESM2_15B_HF_ID,
    esm_hidden_size,
    sample_attention_activations,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ESM-2 activations used for ProtSAE training."
    )
    parser.add_argument("--data-root", default="deepgo_dataset")
    parser.add_argument("--output-root", default="activation_dataset")
    parser.add_argument("--esm-model", default=ESM2_15B_HF_ID)
    parser.add_argument("--layers", nargs="+", type=int, default=[35])
    parser.add_argument("--sample-num", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_zarr_store(save_dir, layers, batch_size, sample_num, hidden_size):
    save_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(save_dir / "dataset.zarr"))
    root = zarr.group(store=store, overwrite=True)
    object_codec = numcodecs.JSON()
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
    )

    root.create_dataset(
        "entry",
        shape=(0,),
        chunks=(batch_size,),
        dtype=object,
        object_codec=object_codec,
        overwrite=True,
    )
    root.create_dataset(
        "tags",
        shape=(0,),
        chunks=(batch_size,),
        dtype=object,
        object_codec=object_codec,
        overwrite=True,
    )
    for layer in layers:
        root.create_dataset(
            f"layer_{layer}",
            shape=(0, sample_num, hidden_size),
            chunks=(batch_size, sample_num, hidden_size),
            dtype=np.float16,
            compressor=compressor,
            overwrite=True,
        )
    return root, store


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    model = AutoModel.from_pretrained(
        args.esm_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )
    model.eval()
    hidden_size = esm_hidden_size(model)

    output_root = Path(args.output_root) / "esm2_15b_deepgo_dataset"
    for ont in ONTOLOGIES:
        for split in ("train", "valid", "test"):
            split_name = f"{ont}_{split}"
            dataset = load_deepgo_sequence_dataset(
                data_root=args.data_root,
                ont=ont,
                split=split,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
            root, store = create_zarr_store(
                save_dir=output_root / split_name,
                layers=args.layers,
                batch_size=args.batch_size,
                sample_num=args.sample_num,
                hidden_size=hidden_size,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_protein_batch,
                pin_memory=True,
            )

            for batch in tqdm(dataloader, desc=f"Preparing {split_name}"):
                layer_outputs = sample_attention_activations(
                    model=model,
                    batch=batch,
                    layers=args.layers,
                    sample_num=args.sample_num,
                )
                root["entry"].append(np.asarray(batch["entry"], dtype=object))
                root["tags"].append(np.asarray(batch["tags"], dtype=object))
                for layer in args.layers:
                    root[f"layer_{layer}"].append(layer_outputs[layer])

            zarr.consolidate_metadata(store)


if __name__ == "__main__":
    main()

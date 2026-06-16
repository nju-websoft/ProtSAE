import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from protsae.data import load_normal_forms, load_terms
from protsae.esm import (
    ESM2_15B_HF_ID,
    attention_output_module,
    decode_protein_sequence,
    masked_sequence_positions,
    model_input_device,
    set_random_seed,
)
from protsae.model import ProtSAE
from protsae.ontology import Ontology


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ESM-2 inference with ProtSAE inserted into a hidden layer."
    )
    parser.add_argument("--data-root", default="deepgo_dataset")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--esm-model", default=ESM2_15B_HF_ID)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ont", choices=["bp", "cc", "mf"], default="cc")
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=None)
    parser.add_argument("--input-dim", type=int, default=5120)
    parser.add_argument("--go-id", default=None)
    parser.add_argument("--intervention-scale", type=float, default=0.2)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-strategy", choices=["low_activation", "random"], default="low_activation")
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_num_features(ont):
    return 40000 if ont == "bp" else 30000


def concept_indices_for_go_id(data_root, ont, go_id, terms_dict):
    ontology = Ontology(Path(data_root) / "go.obo", with_rels=True)
    return [
        terms_dict[ancestor]
        for ancestor in ontology.ancestors(go_id)
        if ancestor in terms_dict
    ]


def load_protsae(args):
    if args.num_features is None:
        args.num_features = default_num_features(args.ont)

    _, terms_dict = load_terms(args.data_root, args.ont)
    nf1, nf2, nf3, nf4, relations, zero_concepts = load_normal_forms(
        Path(args.data_root) / "go.norm",
        terms_dict,
    )
    del nf1, nf2, nf3, nf4

    model = ProtSAE(
        input_dim=args.input_dim,
        num_concepts=len(terms_dict),
        num_features=args.num_features,
        top_k=args.top_k,
        num_zero_concepts=len(zero_concepts),
        num_relations=len(relations),
        dtype=torch.float32,
        seed=args.seed,
        device=args.device,
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    return model, terms_dict


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmForMaskedLM.from_pretrained(
        args.esm_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )
    esm_model.eval()

    protsae, terms_dict = load_protsae(args)
    target_concepts = None
    if args.go_id:
        target_concepts = concept_indices_for_go_id(
            data_root=args.data_root,
            ont=args.ont,
            go_id=args.go_id,
            terms_dict=terms_dict,
        )
        if not target_concepts:
            raise ValueError(f"GO term {args.go_id} is not available in {args.ont}.")

    insert_module = attention_output_module(esm_model, args.layer)

    def protsae_hook(_, __, hidden_states):
        x_hat = protsae.reconstruct(
            hidden_states.to(device=args.device, dtype=torch.float32),
            concept_indices=target_concepts,
            intervention_scale=args.intervention_scale if target_concepts else 0.0,
        )
        return x_hat.to(device=hidden_states.device, dtype=hidden_states.dtype)

    input_df = pd.read_csv(args.input_csv)
    rows = []
    for row in tqdm(input_df.itertuples(index=False), total=len(input_df), desc="ProtSAE inference"):
        sequence = getattr(row, args.sequence_column)
        sequence_id = getattr(row, args.id_column) if args.id_column else len(rows)
        input_ids, attention_mask, mask_positions = masked_sequence_positions(
            model=esm_model,
            tokenizer=tokenizer,
            sequence=sequence,
            layer=args.layer,
            mask_ratio=args.mask_ratio,
            max_length=args.max_length,
            strategy=args.mask_strategy,
        )

        masked_ids = input_ids.clone()
        masked_ids[0, mask_positions] = tokenizer.mask_token_id

        hook = insert_module.register_forward_hook(protsae_hook)
        try:
            with torch.no_grad():
                outputs = esm_model(
                    input_ids=masked_ids.to(model_input_device(esm_model)),
                    attention_mask=attention_mask.to(model_input_device(esm_model)),
                )
        finally:
            hook.remove()

        predicted_tokens = torch.argmax(outputs.logits[0, mask_positions], dim=-1)
        generated_ids = input_ids.clone()
        generated_ids[0, mask_positions] = predicted_tokens

        rows.append(
            {
                "id": sequence_id,
                "go_id": args.go_id,
                "original_sequence": sequence,
                "masked_sequence": decode_protein_sequence(tokenizer, masked_ids[0]),
                "generated_sequence": decode_protein_sequence(tokenizer, generated_ids[0]),
                "mask_positions": ",".join(str(int(pos)) for pos in mask_positions.detach().cpu()),
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

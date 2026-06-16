# ProtSAE

Implementation for **ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders**.

1. prepare ESM-2 activations for training,
2. train ProtSAE with annotation and ontology guidance,
3. insert a trained ProtSAE into ESM-2 for reconstruction or GO-guided inference.


## Code Layout

```text
.
├── prepare_protsae_activations.py   # extract ESM-2 hidden activations
├── train_protsae.py                 # train ProtSAE
├── infer_protsae.py                 # insert ProtSAE into ESM-2 inference
├── protsae/
│   ├── model.py                     # ProtSAE model
│   ├── data.py                      # DeepGO activation and ontology axiom loading
│   ├── esm.py                       # ESM-2 hooks and masking utilities
│   ├── ontology.py                  # minimal GO parser for inference steering
│   └── torch_utils.py               # tensor dataloader
└── requirements.txt
```

## Data

ProtSAE uses the DeepGO2-style protein function prediction dataset built from experimentally supported Swiss-Prot annotations.

- Public processed training data source: <https://github.com/bio-ontology-research-group/deepgo2>

Expected structure:

```text
deepgo_dataset/
├── bp/
│   ├── train_data.pkl
│   ├── valid_data.pkl
│   ├── test_data.pkl
│   └── terms.pkl
├── cc/
│   ├── train_data.pkl
│   ├── valid_data.pkl
│   ├── test_data.pkl
│   └── terms.pkl
├── mf/
│   ├── train_data.pkl
│   ├── valid_data.pkl
│   ├── test_data.pkl
│   └── terms.pkl
├── go.norm
└── go.obo
```

The training script uses `train_data.pkl`, `valid_data.pkl`, `terms.pkl`, and `go.norm`. The test split is kept in the dataset layout for compatibility with the original benchmark.

## Environment

```bash
pip install -r requirements.txt
```

The default PLM is <https://huggingface.co/facebook/esm2_t48_15B_UR50D>. You can also pass a local Hugging Face snapshot path via `--esm-model`.

ESM-2 15B is large. Activation preparation and insertion inference normally require multiple high-memory GPUs or a working `device_map=auto` setup.

## Step 1: Prepare Training Activations

ProtSAE trains on sampled token activations from an ESM-2 hidden layer. The paper uses ESM2-15B and layer 35 for the main function-guided ProtSAE.

```bash
python prepare_protsae_activations.py \
  --data-root deepgo_dataset \
  --output-root activation_dataset \
  --layers 35 \
  --sample-num 100 \
  --batch-size 48
```

Output:

```text
activation_dataset/
└── esm2_15b_deepgo_dataset/
    ├── bp_train/dataset.zarr
    ├── bp_valid/dataset.zarr
    ├── bp_test/dataset.zarr
    ├── cc_train/dataset.zarr
    └── ...
```

## Step 2: Train ProtSAE

Default released settings:

- `layer = 35`
- `top_k = 1000`
- `num_features = 40000` for BPO
- `num_features = 30000` for MFO and CCO
- `lambda_annot = 10`
- `lambda_axiom = 1`
- `learning_rate = 5e-4`

Example for BPO:

```bash
python train_protsae.py \
  --data-root deepgo_dataset \
  --activation-root activation_dataset \
  --ont bp \
  --layer 35 \
  --top-k 1000 \
  --output-dir final_result
```

Outputs:

```text
final_result/
├── protsae_bp_layer35_topk1000.pt
└── protsae_bp_layer35_topk1000.json
```

The checkpoint is selected using validation `L_annot` and normalized reconstruction loss after `--save-after-epoch`.

## Step 3: Insert ProtSAE During ESM-2 Inference

`infer_protsae.py` registers a forward hook on an ESM-2 attention-output module and replaces the hidden state with the ProtSAE reconstruction.

Without `--go-id`, the script performs plain ProtSAE reconstruction. With `--go-id`, it boosts the corresponding `z_def` features for the GO term and its ancestors before decoding.

Input CSV requirements:

- a sequence column, default `Sequence`;
- optional ID column.

Example:

```bash
python infer_protsae.py \
  --data-root deepgo_dataset \
  --input-csv examples/input_sequences.csv \
  --output-csv outputs/protsae_generated.csv \
  --checkpoint final_result/protsae_cc_layer35_topk1000.pt \
  --ont cc \
  --layer 35 \
  --top-k 1000 \
  --go-id GO:0009570 \
  --mask-ratio 0.3 \
  --mask-strategy low_activation
```

Output CSV columns:

- `id`
- `go_id`
- `original_sequence`
- `masked_sequence`
- `generated_sequence`
- `mask_positions`


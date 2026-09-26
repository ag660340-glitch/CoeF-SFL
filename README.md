# Low-frequency Split Federated Learning — method code

Configurations and the implementation of each training method and curvature approximation used in the paper.
Backbones, data loading, and generic utilities are omitted, so this excerpt is meant for reading, not for running end to end.

## Configurations (`configs/`)

- `common.args`: shared protocol. 50 clients, 10% participation per round, 100 rounds (3 warm-up), 1 local epoch, batch 32, LoRA r=4, alpha=8 on all linear layers, no LR schedule.
- `splits.args`: IID, or label-Dirichlet non-IID (alpha 0.5 for DistilRoBERTa, alpha 0.1 for ViT-Tiny).
- `data/*.args`: per-dataset arguments (sequence length, QA stride).
- `arms/*.args`: per-method arguments. `${LR}` is replaced by the learning rate.

Tuning: each arm is run with seed 123 at learning rates {1e-3, 5e-4, 1e-4}. The best seed-123 learning rate is then run with seeds 1234 and 12345, and the median over the three seeds is reported.
MU-SplitFed runs stop early at round r >= 50 if the test metric is identical at rounds r-20, r-10 and r, and that value is reported.

## Methods

| arm | name in paper | files |
|---|---|---|
| vanilla | Vanilla SFL | `splitfed_vanilla_hom.py`, `train/train_fl.py` |
| stale, upperdiag | — | `splitfed_hess_diag.py`, `train/train_fl_hess_diag.py` |
| rsloc_MpJ | CoeF-J | `splitfed_hess_diag.py`, `train/train_fl_hess_diag.py`, `train/hd_omega.py` |
| dcasgd | CoeF-D | `splitfed_kprobe.py`, `train/hd_kprobe.py` |
| fisher | GGN (low-rank, Lanczos) | `splitfed_hess_diag.py`, `train/train_fl_hess_diag.py` |
| hutchinson | Diagonal (Hutchinson) | `splitfed_hess_diag.py`, `train/train_fl_hess_diag.py` |
| gas | GAS | `splitfed_hess_diag.py`, `train/hd_gas.py` |
| acc, acc_* | AccSFL (+ auxiliary-net ablations) | `splitfed_acc_homo.py`, `train/train_fl.py`, `model/auxnet.py` |
| fsl_sage | FSL-SAGE | `splitfed_fsl_sage.py`, `train/train_fl_fsl_sage.py`, `train/train_fl_cse_fsl.py` (auxiliary-net builder) |
| mu_splitfed | MU-SplitFed | `splitfed_mu_splitfed.py`, `train/train_fl_mu_splitfed.py` |

- `splitfed_*.py` are the per-round drivers (client sampling, exchange, aggregation).
- `train/` holds the client and server local updates and the correction and approximation code.
- `train/train_fl.py` is an excerpt that keeps only the vanilla SFL and AccSFL local updates.
- AccSFL auxiliary-net ablations (`model/auxnet.py`):
  - `acc_mlp`, `acc_mlp4`, `acc_mlp8`, `acc_mlp12`: 1, 4, 8, 12 hidden layers of width d/2.
  - `acc_lora`, `acc_lora2`: FC with LoRA, r=4/alpha=8 and r=2/alpha=4.

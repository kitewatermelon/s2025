#!/usr/bin/env bash
# scripts/run_vqvae_ablation.sh
#
# VQ-GAN ablation — 3 GPUs in parallel
#
# GPU 4: ratio-2  VQ-GAN CBCT → CT  (64×64 latent, 1 stride-2)
# GPU 5: ratio-8  VQ-GAN CBCT → CT  (16×16 latent, 3 stride-2)
# GPU 6: baseline VQ-GAN CBCT → CT  (32×32 latent, 2 stride-2)
#         → then n=5 VDM training (slice ablation)
#
# Depth-compression math (D_latent=1 required for VDM .squeeze(2)):
#   ratio-2: D=3  →(s2)→ 1               VDM num_slices=3
#   ratio-4: D=7  →(s2)→ 3 →(s2)→ 1     VDM num_slices=7  [baseline]
#   ratio-8: D=9  →(s2)→ 4 →(s2)→ 2 →(s2)→ 1  VDM num_slices=9
#
# Usage:
#   bash scripts/run_vqvae_ablation.sh          # launch all 3 in parallel
#   bash scripts/run_vqvae_ablation.sh gpu4     # GPU 4 only
#   bash scripts/run_vqvae_ablation.sh gpu5     # GPU 5 only
#   bash scripts/run_vqvae_ablation.sh gpu6     # GPU 6 only

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

LOG_DIR="$PROJ/logs/vqvae_ablation"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y%m%d_%H%M%S'; }

# ── GPU 4: ratio-2 VQ-GAN ─────────────────────────────────────────────────────
run_gpu4() {
    local ts; ts=$(timestamp)
    echo "[GPU4] $(date) — ratio-2 CBCT VQ-GAN start"
    python train_vqgan.py -c configs/comp_abl/ae_ratio2_cbct.yaml \
        2>&1 | tee "$LOG_DIR/gpu4_ratio2_cbct_${ts}.log"

    echo "[GPU4] $(date) — ratio-2 CT VQ-GAN start"
    python train_vqgan.py -c configs/comp_abl/ae_ratio2_ct.yaml \
        2>&1 | tee "$LOG_DIR/gpu4_ratio2_ct_$(timestamp).log"
    echo "[GPU4] $(date) — ratio-2 COMPLETE"
}

# ── GPU 5: ratio-8 VQ-GAN ─────────────────────────────────────────────────────
run_gpu5() {
    local ts; ts=$(timestamp)
    echo "[GPU5] $(date) — ratio-8 CBCT VQ-GAN start"
    python train_vqgan.py -c configs/comp_abl/ae_ratio8_cbct.yaml \
        2>&1 | tee "$LOG_DIR/gpu5_ratio8_cbct_${ts}.log"

    echo "[GPU5] $(date) — ratio-8 CT VQ-GAN start"
    python train_vqgan.py -c configs/comp_abl/ae_ratio8_ct.yaml \
        2>&1 | tee "$LOG_DIR/gpu5_ratio8_ct_$(timestamp).log"
    echo "[GPU5] $(date) — ratio-8 COMPLETE"
}

# ── GPU 6: baseline VQ-GAN → n=5 VDM ─────────────────────────────────────────
# Step 1: train baseline ratio-4 CBCT VQ-GAN  →  checkpoints/2.5D_cbct_PATCH_NORM/
# Step 2: train baseline ratio-4 CT VQ-GAN    →  checkpoints/2.5D_ct_PATCH_NORM/
# Step 3: n=5 VDM (slice ablation) using those checkpoints
# CUDA_VISIBLE_DEVICES=6 maps GPU 6 → cuda:0 for Accelerator in step 3;
# steps 1-2 use device=6 set in the config itself.
run_gpu6() {
    local ts; ts=$(timestamp)
    echo "[GPU6] $(date) — baseline CBCT VQ-GAN start"
    python train_vqgan.py -c configs/baseline/ae_cbct.yaml \
        2>&1 | tee "$LOG_DIR/gpu6_baseline_cbct_${ts}.log"

    echo "[GPU6] $(date) — baseline CT VQ-GAN start"
    python train_vqgan.py -c configs/baseline/ae_ct.yaml \
        2>&1 | tee "$LOG_DIR/gpu6_baseline_ct_$(timestamp).log"

    echo "[GPU6] $(date) — n=5 VDM (slice ablation) start"
    CUDA_VISIBLE_DEVICES=6 python train_vdm_uvit_2.5d.py \
        -c configs/slice_abl/n5.yaml \
        2>&1 | tee "$LOG_DIR/gpu6_slice_abl_n5_$(timestamp).log"
    echo "[GPU6] $(date) — GPU6 COMPLETE"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
TARGET="${1:-all}"

case "$TARGET" in
    gpu4) run_gpu4 ;;
    gpu5) run_gpu5 ;;
    gpu6) run_gpu6 ;;
    all)
        echo "=== Starting VQ-GAN ablation on GPUs 4, 5, 6 ==="
        run_gpu4 &  PID4=$!
        run_gpu5 &  PID5=$!
        run_gpu6 &  PID6=$!
        echo "  GPU4 PID=$PID4  (ratio-2 VQ-GAN)"
        echo "  GPU5 PID=$PID5  (ratio-8 VQ-GAN)"
        echo "  GPU6 PID=$PID6  (baseline VQ-GAN + n=5 VDM)"
        echo "Logs: $LOG_DIR"

        wait $PID4; echo "[GPU4] EXIT $?"
        wait $PID5; echo "[GPU5] EXIT $?"
        wait $PID6; echo "[GPU6] EXIT $?"
        echo "=== All done ==="
        ;;
    *)
        echo "Usage: $0 [all|gpu4|gpu5|gpu6]"
        exit 1
        ;;
esac

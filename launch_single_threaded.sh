#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin_base_patch4_window7_224.yaml --zip --data-path "${INPUT_DIR}" --cache-mode no --accumulation-steps 8 --batch-size 32 --pyramid_adversarial_training="${PYRAMID_ADVERSARIAL_TRAINING}" --output_dir="${OUTPUT_DIR}"
#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_base_patch4_window7_224.yaml --zip --data-path "${INPUT_DIR}" --cache-mode ${CACHE_MODE} --accumulation-steps 2 --batch-size 64 --pyramid_adversarial_training=${PYRAMID_ADVERSARIAL_TRAINING} --output_dir="${OUTPUT_DIR}"
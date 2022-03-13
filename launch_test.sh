# Run on a much smaller version of the dataset.
#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py --cfg configs/swin_base_patch4_window7_224.yaml --zip --data-path data/ImageNet-Zip-Test --cache-mode no --accumulation-steps 4 --batch-size 64
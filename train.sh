# export no_proxy=localhost
# tensorboard --logdir hfml/YOLOX_outputs/hfml/tensorboard/ &
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python3 hfml/train.py -b 192 --experiment-name hfml3 --fp16 -f hfml/exp.py
# CUDA_VISIBLE_DEVICES=1,2 python3 hfml/train.py -b 48 -f hfml/exp.py
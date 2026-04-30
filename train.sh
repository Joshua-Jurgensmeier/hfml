# export no_proxy=localhost
# tensorboard --logdir hfml/YOLOX_outputs/hfml/tensorboard/ &
python3 hfml/train.py -b 128 --experiment-name hfml  --fp16 -f hfml/exp.py
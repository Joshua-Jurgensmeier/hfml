# High-Frequency Machine-Learning (HFML)
The HFML project is a YOLO-based signal detector and classifier for Radio Frequency (RF) signals in the High Frequency (HF) band (3-30MHz) designed to run on the Nvidia Jetson Orin Nano with an RTL-SDR Blog v4.
## Nano Setup
The docker image defined by `Dockerfile.nano` contains all of the setup necessary for inference. Docker must be installed and configured on the Nano. It's recommended to follow a setup procedure like this: https://www.jetson-ai-lab.com/tutorials/ssd-docker-setup/. Once setup, run
```
docker build -t hfml -f Dockerfile.nano .
```
There's a bit of configuration that must be completed on the host. Refer to the "Getting Started on Linux" section at https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/.

Additionally, you need to make sure you can access the USB bus. This is one way to make that happen:
```
sudo usermod -aG plugdev $USER
```

`devcontainer.json` contains a few commented out lines related to enabling USB support. These are initially commented out for security when using the devcontainer on a training server. To access the RTL-SDR over USB on the Nano from inside the devcontainer, uncomment them.

## Run
On the Nano, either run the devcontainer with VS Code, or manually launch the container with this annoyingly long command after substituting `/path/to/hfml` with the path to your HFML repo:
```
docker run -it --gpus all --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=512M --net=host --privileged --rm -v /dev/bus/usb:/dev/bus/usb -v /path/to/hfml:/home/vscode/hfml hfml
```

To detect/classify live signals from an RTL-SDR, within the container run
```
cd /home/vscode/hfml
python hfml/run.py -f hfml/exp.py -c hfml.pth --device gpu --save_result --fc 13e6
```

## Train
Dataset and training steps WIP
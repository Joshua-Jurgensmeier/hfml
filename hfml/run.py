# Copyright Joshua Jurgensmeier 2026

import datetime
from pathlib import Path
import time
import argparse
import os

from rtlsdr import RtlSdr
from scipy.io import savemat
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from yolox.exp import get_exp
from yolox.utils import get_model_info

from dsp import EnergyDetector, Spectrogrammer
from core import HFML, Predictor, HFML_CLASSES

def make_parser():
    parser = argparse.ArgumentParser("HFML Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fc", default=13e6, type=float, help="Center Frequency")
    return parser

def get_sdr(fs: int, fc: int):
    sdr = RtlSdr()
    sdr.sample_rate = fs
    sdr.set_bias_tee(True)
    sdr.gain=20
    sdr.fc = fc

    return sdr

def main(exp, args):
    # SDR
    FRAMES = 16
    fs = int(2**21)
    frame_s = 1
    s = frame_s*FRAMES
    N = fs*s
    N_frame = frame_s * fs
    TOTAL_BW = fs * 13
    min_f = int(3e6)
    max_f = int(30e6)

    # Spectrogram
    M = 128
    NFFT = fs // 16
    hop_factor = 0.25
    hop_length = int(NFFT * hop_factor)
    clamp_dB = 20

    # Energy Detector
    n_fbins = 64
    pwr_thresh_dB = 6
    signal_width_Hz = 512
    signal_len_s = 0.5
    win_kernel_size = (signal_width_Hz // (fs // NFFT), int( (N//hop_length) / (FRAMES*frame_s) * signal_len_s))

    # NFFT_chan = NFFT // M
    # print(f"{NFFT=}")
    # print(f"{hop_length=}")
    # print("Time resolution: ", NFFT / fs)
    # print("Freq resolution: ", fs // NFFT)
    # print(f"Dimensions (px) txf:  {int(N / hop_length)} x {NFFT_chan}")
    # print(f"Dimensions (ph) sxf KHz:  {s} x {fs_chan * 1e-3}")
    # print(f"2.4KHz percent of channelized width:{2400/fs_chan*100}%" )
    # print(f"{win_kernel_size=}")

    ed = EnergyDetector(n_fbins, pwr_thresh_dB, win_kernel_size)
    spg = Spectrogrammer(NFFT, hop_factor, M)
    sdr = get_sdr(fs, args.fc)
    model = exp.get_model()
    
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")

    cmplx_dtype = torch.complex64
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
            cmplx_dtype = torch.complex32
    model.eval()

    predictor = Predictor(model, exp, HFML_CLASSES, args.device, args.fp16)

    hfml = HFML(ed, spg, predictor)

    vis_folder = "out"
    os.makedirs(vis_folder, exist_ok=True)
    cap_id = 0

    # Output config
    print(f"Model Summary: {get_model_info(model, exp.test_size)}")

    while True:

        x_frame = []
        for i in range(FRAMES):
            x_frame.append(sdr.read_samples(N_frame))  # Get samples
        
        x = torch.tensor(np.concatenate(x_frame, axis=-1), dtype=cmplx_dtype)

        outputs, img_infos = hfml.inference(x)
        print("Finshed processing")

        print("visualizing")
        if outputs is not None:
            for output, img_info in zip(outputs,img_infos):
                if output[0] is not None:
                    result_image = hfml.visual(output[0], img_info, predictor.confthre)
                    if args.save_result:
                        print("Writing result")
                        save_file_name = os.path.join(vis_folder, f"cap_{cap_id}.png")
                        cv2.imwrite(save_file_name, result_image)
                    else:
                        cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                        cv2.imshow("yolox", result_image)
                        ch = cv2.waitKey(0)
                        if ch == 27 or ch == ord("q") or ch == ord("Q"):
                            break
                    cap_id = cap_id + 1

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    main(exp, args)

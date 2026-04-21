import torch
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis
import time

HFML_CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7")

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

class HFML:
    def __init__(self, energy_detector, spectrogrammer, predictor: Predictor, clamp_dB=20):
        self.ed = energy_detector
        self.spgr = spectrogrammer
        self.predictor = predictor
        self.clamp_dB = clamp_dB

    def inference(self, x):
        with torch.no_grad():
            sgs = self.spgr(x)
            detects, noise_floor_dB = self.ed(sgs)
            noise_floor_dB = noise_floor_dB.mean()
            if detects.any():
                detected_sgs = sgs[detects]
                imgs: torch.Tensor = detected_sgs.clamp(max=noise_floor_dB+self.clamp_dB)
                imgs = imgs.cpu().numpy()
                outputs, img_info = self.predictor.inference(imgs)
                return outputs, img_info
            else:
                return None, None
        
    def visual(self, output, img_info, cls_conf=0.35):
        return self.predictor.visual(output, img_info, cls_conf)
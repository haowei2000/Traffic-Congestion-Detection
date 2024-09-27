import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:

    def __init__(self):
        # 初始化参数
        self.img_size = 640  # 输入图像大小
        self.threshold = 0.3  # 检测阈值
        self.stride = 1  # 步幅

        self.weights = './weights/yolov5m.pt'  # 模型权重文件路径

        # 选择设备（GPU或CPU）
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)

        # 加载模型
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()  # 使用半精度

        self.m = model
        # 获取模型的类别名称
        self.names = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):
        # 预处理输入图像
        img0 = img.copy()  # 复制原始图像
        img = letterbox(img, new_shape=self.img_size)[0]  # 调整图像大小
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB并调整维度
        img = np.ascontiguousarray(img)  # 转换为连续数组
        img = torch.from_numpy(img).to(self.device)  # 转换为Tensor并移动到设备
        img = img.half()  # 使用半精度
        img /= 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加一个维度

        return img0, img

    def detect(self, im):
        # 检测图像中的目标
        im0, img = self.preprocess(im)  # 预处理图像

        pred = self.m(img, augment=False)[0]  # 模型预测
        pred = pred.float()  # 转换为浮点数
        pred = non_max_suppression(pred, self.threshold, 0.4)  # 非极大值抑制

        boxes = []
        for det in pred:
            if det is not None and len(det):
                # 调整坐标到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]  # 获取类别名称
                    if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                        continue  # 过滤不需要的类别
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append((x1, y1, x2, y2, lbl, conf))  # 添加检测框信息

        return boxes

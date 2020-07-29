# ==============================================================================
# Copyright 2019 Florent Mahoudeau.
# Licensed under the MIT License.
# ==============================================================================
import time
import argparse
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor

import sys
sys.path.insert(0, '../')

# 内部包
from shelfnet.config import cfg
from shelfnet.config import update_config
from shelfnet import models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test speed keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--image',
                        help='image path',
                        type=str,
                        default=None)

    args = parser.parse_args()
    return args


def predict():
    args = parse_args()
    update_config(cfg, args)

    print('Compute device: ' + "cuda:%d" %
          cfg.GPUS[0] if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:%d" %
                          cfg.GPUS[0] if torch.cuda.is_available() else "cpu")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # count parameter number
    model = models.shelfnet.get_pose_net(cfg, is_train=False)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)

    model = model.to(device)
    model.eval()
    run_time = list()
    print("Input shape = " + str(cfg.MODEL.IMAGE_SIZE[::-1]))

    # 加载图像
    img = Image.open(args.image)
    img = img.resize(*cfg.MODEL.IMAGE_SIZE[::-1])
    img = ToTensor()(img)
    _ = model(img.to(device))

    run_time = time.time()
    with torch.no_grad():
        for i in range(0, 1000):
            outputs = model(img.to(device))

    run_time = time.time() - run_time
    print("Out:\n", outputs)
    print('Mean running time is {:.5f}'.format(run_time/1000))
    print('FPS = {:.1f}'.format(1000 / run_time))


if __name__ == '__main__':
    predict()

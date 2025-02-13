import argparse
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import tools._init_paths
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.pose_recognition.lib.config import cfg
from src.pose_recognition.lib.config import update_config
from src.pose_recognition.lib.models.pose_hrnet import *
# from core.loss import JointsMSELoss
# from core.loss import DepthLoss
# from core.loss import hoe_diff_loss
# from core.loss import Bone_loss

# from core.function import train
# from core.function import validate

# from utils.utils import get_optimizer
# from utils.utils import save_checkpoint
# from utils.utils import create_logger
# from utils.utils import get_model_summary
import models
from PIL import Image

def parse_args(*args):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='src/pose_recognition/experiments/coco/segm-4_lr1e-3.yaml',
                        type=str)
    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='real_brt/models')
    
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    parser.add_argument('--device', default='cpu')
    parser.add_argument('img_path', nargs='?', default='real_brt/demo3.jpg')

    a = parser.parse_args(args)

    return a


def hboe(person):
    args = parse_args()
    update_config(cfg, args)

    # logger, _, _ = create_logger(
    #     cfg, args.cfg, 'valid')

    # logger.info(pprint.pformat(args))
    # logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model = eval('src/pose_recognition/models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # ).to(args.device)
    model = get_pose_net(cfg , is_train= False)
    # logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load('src/pose_recognition/models/model_hboe.pth', map_location=torch.device(args.device)), strict=True)

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    img = person
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256))

    input = transform(img).unsqueeze(0)
    input = input.float()

    model.eval()
    _ , hoe_output = model(input)
    ori = torch.argmax(hoe_output[0]) * 5
    print("The predicted orientation angle is: {}".format(ori))

    return format(ori)

    # illustrate hoe_output
    # import matplotlib.pyplot as plt
    # for i in range(hoe_output.shape[1]):
    #     plt.scatter(i * 5, hoe_output[0, i].detach().numpy())
    # plt.savefig("plot.png")

if __name__ == '__main__':
    import numpy as np
    image = np.full(shape=(360,640),fill_value=255,dtype=np.uint8)
    hboe(image)

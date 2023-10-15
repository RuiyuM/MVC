import os
# import timm
import tome
import sys
import torch
import random
import warnings
import pickle
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torch.optim import AdamW
# from timm.data import Mixup
from timm.models import create_model
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from timm.utils import NativeScaler, get_state_dict, ModelEma

from timm.models.vision_transformer import _create_vision_transformer
sys.dont_write_bytecode = True
warnings.filterwarnings('ignore')
from unlabeled_Sampling_Dataset import Unlabeled_Dataset
import utils as tool
import parser_test as parser_2
from model.svcnn import SVCNN
from model.mvcnn_new import MVCNNNew
from model.gvcnn import GVCNN
from model.dan import DAN
from model.cvr import CVR
from model.mvfn import MVFN
from model.smvcnn import SMVCNN
from model.MVT import MVT

from dataset_single_view import SingleViewDataset
from dataset_multi_view import MultiViewDataset
from loss import LabelCrossEntropy
from engine_single_view import SingleViewEngine
from validation_engine_for_validation import MultiViewEngine
import sampling
from dataset import MVDataSet
from data_set_object_wise import object_wise_dataset
from unlabeled_dataset_object_wise import unlabeled_object_wise_dataset
import torchvision.transforms as transforms
from transforms import *





if __name__ == '__main__':
    # set options
    opt = parser_2.get_parser()
    train_txt = 'v2_trainmodel40.txt'
    test_txt = 'v2_testmodel40.txt'
    img_ext = 'png'
    views_number = opt.MAX_NUM_VIEWS
    # if opt.DATA_SET == 'M40v2':
    opt.nb_classes = 40  # ModelNet40 class number
    # opt.num_validation_view = 2
    train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875])])
    normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    print("random_________________")
    # print(train_dataset.selected_ind_train)
    for model_idx in [3,4,5,6,7,8,9]:
        for num_validation_view in [2]:
            # file_path = f'./weight_mv/MVT_query:{model_idx}_M40v2_patch_based_selection_476.pt'
            file_path = f'./weight_mv/MVT_query:{model_idx}_M40v2_random_476.pt'

            model_stage2 = create_model('vit_small_patch16_224', pretrained=False)
            num_classes = 40
            model_stage2.head = torch.nn.Linear(model_stage2.head.in_features, num_classes)
            # Ensure tome.patch.timm is defined or imported
            tome.patch.timm(model_stage2)
            model_stage2.r = 0

            model_stage2.load_state_dict(torch.load(file_path, map_location="cuda:0"))

            opt.num_validation_view = num_validation_view

            # Re-initialize your dataset and DataLoader here if they depend on opt.num_validation_view
            valid_dataset = object_wise_dataset("", test_txt, 40, 'valid',
                                                image_tmpl="_{:03d}." + img_ext,
                                                max_num_views=views_number,
                                                num_validation=opt.num_validation_view,
                                                validation_mode=True,
                                                transform=torchvision.transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                ]))
            valid_data = DataLoader(valid_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                    shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)

            # If other parts of your code depend on the changed opt values, update them here as well...

            engine = MultiViewEngine(model_stage2, None, valid_data, 40, None, None, None,
                                     opt.MV_WEIGHT_PATH, "cuda:0", opt.MV_TYPE, None, 0,
                                     None)
            acc = engine.train_base(1)
            print(f"Model Index: {model_idx}, Num Validation View: {num_validation_view}, Accuracy: {100 * acc}")
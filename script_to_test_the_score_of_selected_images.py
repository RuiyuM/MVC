import os
# import timm
import tome
import sys
from torch.autograd import Variable
import torch
import random
import warnings
from scipy.optimize import linear_sum_assignment
import itertools
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
from engine_multi_view import MultiViewEngine
import sampling
from dataset import MVDataSet
from data_set_object_wise import object_wise_dataset
from unlabeled_dataset_object_wise import unlabeled_object_wise_dataset
import torchvision.transforms as transforms
from transforms import *
# 'modelnet40v2png_ori4/airplane/test/airplane_0627'

def calcualte_cost(label_metrics, training_metrics):
    cost_matrix = torch.mm(label_metrics.t(), training_metrics)  # 196*196
    # cost_matrix = torch.mm(label_metrics, training_metrics.t())
    cost_matrix_np = cost_matrix.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    total_cost = cost_matrix_np[row_ind, col_ind].sum()
    return total_cost

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[1])


if __name__ == '__main__':
    list_file = 'v2_testmodel40.txt'
    video_list = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]

    for video in video_list:
        image_paths = [f'{video.path}_001.png', f'{video.path}_002.png', f'{video.path}_012.png']
        category = video.path.split('/')[-3]

        train_augmentation = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                                 ])
        data = []
        for image_path in image_paths:
            label = torch.zeros(40)
            label[0] = 1.0  # You might want to modify this according to your label assignment

            image = Image.open(image_path).convert('RGB')
            image = train_augmentation(image)
            data.append((label, image, 1, 0))

        # set options
        opt = parser_2.get_parser()
        train_txt = 'v2_trainmodel40.txt'
        test_txt = 'v2_testmodel40.txt'
        img_ext = 'png'
        views_number = opt.MAX_NUM_VIEWS

        opt.nb_classes = 40  # ModelNet40 class number
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875])])
        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        for model_idx in [3, 9]:
            file_path = f'./weight_mv/MVT_query:{model_idx}_M40v2_random_476.pt'

            model_stage2 = create_model('vit_small_patch16_224', pretrained=False)
            num_classes = 40
            model_stage2.head = torch.nn.Linear(model_stage2.head.in_features, num_classes)

            # Ensure tome.patch.timm is defined or imported
            tome.patch.timm(model_stage2)
            model_stage2.r = 0

            model_stage2.load_state_dict(torch.load(file_path, map_location="cuda:0"))

            model_stage2 = model_stage2.cuda()
            model_stage2.eval()

            label_K_dict = []
            with torch.no_grad():
                for index, (label, image, num_views, object_class) in enumerate(data):
                    inputs = Variable(image).to("cuda:0")
                    targets = Variable(label).to("cuda:0")
                    B, V, C, H, W = 1, 1, 3, 224, 224
                    inputs = inputs.view(-1, C, H, W)

                    outputs, k_metrics, features = model_stage2(B, V, num_views, inputs)
                    label_K_dict.append(k_metrics.squeeze(0))

            cost1 = calcualte_cost(label_K_dict[0], label_K_dict[1])
            cost2 = calcualte_cost(label_K_dict[0], label_K_dict[2])
            cost3 = calcualte_cost(label_K_dict[0], label_K_dict[0])
            print(
                f"Category: {category}, round{model_idx}_similar_images_score: {cost1}####dissimilar_images_score: {cost2}#######self_score: {cost3}")


def apply_function_on_combinations(elements, func):
    """
    Applies a function on all unique combinations of two different elements from a list.

    :param elements: list of elements
    :param func: function to apply on combinations
    """
    # Getting all unique combinations of two elements from the list
    combinations = itertools.combinations(elements, 2)

    # Applying the function on each combination
    results = []
    for combo in combinations:
        result = calcualte_cost(combo[0], combo[1])
        results.append(result)
        print(f"Function applied on {combo}: Result = {result}")

    return results

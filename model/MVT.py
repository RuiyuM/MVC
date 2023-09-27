import timm
import torch.nn as nn
import torch
import inspect
from collections import OrderedDict


class MVT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, max_num_views, num_classes, *args, **kwargs):
        super(MVT, self).__init__(*args, num_classes=num_classes, **kwargs)




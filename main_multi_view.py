import os
# import timm
import tome
import sys
import torch
import random
import warnings
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



def generate_sampling_dataset(selected_ind_train, unselected_ind_train, opt):
    train_txt = 'v2_trainmodel40.txt'
    test_txt = 'v2_testmodel40.txt'
    img_ext = 'png'
    views_number = opt.MAX_NUM_VIEWS
    unlabeled_sampling_labeled_data = selected_ind_train

    unlabeled_sampling_unlabeled_data = unselected_ind_train

    # if opt.DATA_SET == 'M40v2':
    opt.nb_classes = 40  # ModelNet40 class number
    labeled_dataset = unlabeled_object_wise_dataset("", train_txt, 40, 'labeled',

                                                    image_tmpl="_{:03d}." + img_ext,

                                                    max_num_views=views_number,
                                                    view_number=opt.view_num,
                                                    transform=transforms.Compose(
                                                        [transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                                         ]),
                                                    selected_ind_train=unlabeled_sampling_labeled_data,
                                                    unselected_ind_train=unlabeled_sampling_unlabeled_data)
    unlabeled_dataset = unlabeled_object_wise_dataset("", test_txt, 40, 'unlabeled',
                                                      image_tmpl="_{:03d}." + img_ext,

                                                      max_num_views=views_number,
                                                      transform=torchvision.transforms.Compose(
                                                          [transforms.ToTensor(),
                                                           transforms.Normalize(
                                                               mean=[0.485, 0.456,
                                                                     0.406],
                                                               std=[0.229, 0.224, 0.225])
                                                           ]),
                                                      selected_ind_train=unlabeled_sampling_labeled_data,
                                                      unselected_ind_train=unlabeled_sampling_unlabeled_data
                                                      )

    labeled_data = DataLoader(labeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                              shuffle=True,
                              pin_memory=True, worker_init_fn=tool.seed_worker)

    unlabeled_data = DataLoader(unlabeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                shuffle=False,
                                pin_memory=True, worker_init_fn=tool.seed_worker)
    return labeled_data, unlabeled_data



if __name__ == '__main__':
    # set options
    opt = parser_2.get_parser()


    cprint('*' * 25 + ' Start ' + '*' * 25, 'yellow')
    # set seed
    seed = opt.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    # set cudnn
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True

    # set device
    device = torch.device(opt.DEVICE if torch.cuda.is_available() else 'cpu')

    # # define model
    model_stage1 = SVCNN(opt.NUM_CLASSES, opt.ARCHITECTURE, opt.FEATURE_DIM, pretrained=True).to(device)
    # # define criterion
    criterion = LabelCrossEntropy()


    if opt.MV_FLAG == 'TRAIN':
        cprint('*' * 15 + ' Stage 2 ' + '*' * 15, 'yellow')

    # define model
    if opt.MV_TYPE == 'MVCNN_NEW':
        model_stage2 = MVCNNNew(model_stage1).to(device)
    elif opt.MV_TYPE == 'GVCNN':
        model_stage2 = GVCNN(model_stage1, opt.GVCNN_M, opt.ARCHITECTURE, opt.IMAGE_SIZE).to(device)
    elif opt.MV_TYPE == 'DAN':
        model_stage2 = DAN(model_stage1, opt.DAN_H, opt.FEATURE_DIM, opt.DAN_NUM_HEADS_F, opt.DAN_INNER_DIM_F,
                           opt.DAN_DROPOUT_F).to(device)
    elif opt.MV_TYPE == 'CVR':
        model_stage2 = CVR(model_stage1, opt.CVR_K, opt.FEATURE_DIM, opt.CVR_NUM_HEADS_F, opt.CVR_INNER_DIM_F,
                           opt.CVR_NORM_EPS_F,
                           opt.CVR_OTK_HEADS_F, opt.CVR_OTK_EPS_F, opt.CVR_OTK_MAX_ITER_F, opt.CVR_DROPOUT_F,
                           opt.CVR_COORD_DIM_F).to(device)
    elif opt.MV_TYPE == 'MVFN':
        model_stage2 = MVFN(model_stage1, opt.FEATURE_DIM).to(device)
    elif opt.MV_TYPE == 'SMVCNN':
        model_stage2 = SMVCNN(model_stage1, opt.FEATURE_DIM, opt.SMVCNN_D, use_embed=opt.SMVCNN_USE_EMBED).to(device)






    if opt.MV_FLAG in ['TEST', 'CM']:
        model_stage2.load_state_dict(torch.load(opt.MV_TEST_WEIGHT, map_location=device))
        model_stage2.eval()

    # define dataset
    if opt.DATA_SET == 'M40v2':
        train_txt = 'v2_trainmodel40.txt'
        test_txt = 'v2_testmodel40.txt'
        img_ext = 'png'
        views_number = opt.MAX_NUM_VIEWS

        # if opt.DATA_SET == 'M40v2':
        opt.nb_classes = 40  # ModelNet40 class number
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875])])
        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_dataset = object_wise_dataset("", train_txt, 40, 'train',

                                            image_tmpl="_{:03d}." + img_ext,

                                            max_num_views=views_number,
                                            view_number=opt.view_num,
                                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])
                                                                          ]))
        # print(train_dataset.selected_ind_train)
        valid_dataset = object_wise_dataset("", test_txt, 40, 'valid',
                                            image_tmpl="_{:03d}." + img_ext,

                                            max_num_views=views_number,
                                            transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                                                      transforms.Normalize(
                                                                                          mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])
                                                                                      ]))



        train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=True,
                                pin_memory=True, worker_init_fn=tool.seed_worker)

        valid_data = DataLoader(valid_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False,
                                pin_memory=True, worker_init_fn=tool.seed_worker)

    if opt.DATA_SET == 'MVP_N':
        train_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'train', opt.MAX_NUM_VIEWS,
                                         use_train=True)
        # dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # batch = next(iter(dataloader))
        valid_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'valid', opt.MAX_NUM_VIEWS,
                                         use_train=False)

        test_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'test', opt.MAX_NUM_VIEWS,
                                        use_train=False)
        train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=True,
                                pin_memory=True, worker_init_fn=tool.seed_worker)
        valid_data = DataLoader(valid_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False,
                                pin_memory=True, worker_init_fn=tool.seed_worker)
        test_data = DataLoader(test_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False,
                               pin_memory=True, worker_init_fn=tool.seed_worker)
    if opt.MV_FLAG in ['TRAIN', 'TEST']:
        print('Number of Training Sets:', len(train_dataset.data))


    # set path
    if opt.MV_FLAG == 'TRAIN':
        if not os.path.exists(opt.MV_WEIGHT_PATH):
            os.mkdir(opt.MV_WEIGHT_PATH)

    # define engine for other model like DAN
    # engine = MultiViewEngine(model_stage2, train_data, valid_data, opt.NUM_CLASSES, optimizer, scheduler, criterion,
    #                          opt.MV_WEIGHT_PATH, device, opt.MV_TYPE)

    # run multi-view
    for query in tqdm(range(opt.MV_QUERIES)):
        # creat model
        if opt.MV_TYPE == 'MVT':
            model_stage2 = create_model('vit_small_patch16_224', pretrained=True)
            num_classes = 40  # Replace with your number of classes
            model_stage2.head = torch.nn.Linear(model_stage2.head.in_features, num_classes)
            tome.patch.timm(model_stage2)
            model_stage2.r = 0
            # define optimizer
            optimizer = optim.SGD(model_stage2.parameters(), lr=opt.MV_LR_INIT, weight_decay=opt.MV_WEIGHT_DECAY,
                                  momentum=opt.MV_MOMENTUM)
            # optimizer = AdamW(model_stage2.parameters(), lr=opt.MV_LR_INIT, weight_decay=opt.MV_WEIGHT_DECAY)

            scheduler = tool.CosineDecayLR(optimizer, T_max=opt.MV_EPOCHS * len(train_dataset), lr_init=opt.MV_LR_INIT,
                                           lr_min=opt.MV_LR_END, warmup=opt.MV_WARMUP_EPOCHS * len(train_dataset))

        engine = MultiViewEngine(model_stage2, train_data, valid_data, 40, optimizer, scheduler, criterion,
                                 opt.MV_WEIGHT_PATH, device, opt.MV_TYPE)
        if opt.MV_FLAG == 'TRAIN':
            if opt.MV_TYPE in ['MVCNN_NEW', 'GVCNN', 'DAN', 'MVFN', 'SMVCNN', 'MVT']:
                engine.train_base(opt.MV_EPOCHS)
            elif opt.MV_TYPE == 'CVR':
                vert = tool.get_vert(opt.CVR_K)
                engine.train_cvr(opt.MV_EPOCHS, vert, opt.CVR_LAMBDA, opt.CVR_NORM_EPS_F)
        elif opt.MV_FLAG == 'TEST':
            cprint('*' * 10 + ' Valid Sets ' + '*' * 10, 'yellow')
            engine.test(valid_data, opt.TEST_T)
            cprint('*' * 10 + ' Test Sets ' + '*' * 10, 'yellow')
            engine.test(test_data, opt.TEST_T)
        elif opt.MV_FLAG == 'COMPUTATION':
            cprint('*' * 10 + ' Computational Efficiency ' + '*' * 10, 'yellow')
            # define inputs
            inputs = torch.randn(opt.MAX_NUM_VIEWS, 3, opt.IMAGE_SIZE, opt.IMAGE_SIZE).to(device)
            # measure parameters
            p1, p2 = tool.get_parameters(model_stage1, model_stage2)
            # measure FLOPs
            f1, f2 = tool.get_FLOPs(model_stage1, model_stage2, inputs)
            # measure latency
            t1_mean, t1_std, t2_mean, t2_std = tool.get_time(model_stage1, model_stage2, inputs, opt.REPETITION)
            cprint('*' * 10 + ' SVCNN ' + '*' * 10, 'yellow')
            print('Model Size (M):', '%.2f' % (p1 / 1e6))
            print('FLOPs (G):', '%.2f' % (f1 / 1e9))
            print('Latency (ms):', '%.2f' % (t1_mean) + ' (mean)', '%.2f' % (t1_std) + ' (std)')
            cprint('*' * 10 + ' ' + opt.MV_TYPE + ' ' + '*' * 10, 'yellow')
            print('Model Size (M):', '%.2f' % (p2 / 1e6))
            print('FLOPs (G):', '%.2f' % (f2 / 1e9))
            print('Latency (ms):', '%.2f' % (t2_mean) + ' (mean)', '%.2f' % (t2_std) + ' (std)')
        elif opt.MV_FLAG == 'CM':
            cprint('*' * 10 + ' Confusion Matrix ' + '*' * 10, 'yellow')
            confusion_matrix = engine.confusion_matrix(valid_data)
            tool.plot_confusion_matrix(confusion_matrix, opt.GROUPS, opt.MV_TYPE)

        cprint('*' * 25 + ' Finish Training start sampling ' + '*' * 25, 'yellow')
        # define the sampling dataset
        if opt.DATA_SET == 'M40v2':
            unlabeled_sampling_labeled_data = train_dataset.selected_ind_train

            unlabeled_sampling_unlabeled_data = train_dataset.unselected_ind_train


            # if opt.DATA_SET == 'M40v2':
            opt.nb_classes = 40  # ModelNet40 class number
            labeled_dataset = unlabeled_object_wise_dataset("", train_txt, 40, 'labeled',

                                                            image_tmpl="_{:03d}." + img_ext,

                                                            max_num_views=views_number,
                                                            view_number=opt.view_num,
                                                            transform=transforms.Compose(
                                                                [transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(
                                                                     mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                                 ]),
                                                            selected_ind_train=unlabeled_sampling_labeled_data,
                                                            unselected_ind_train=unlabeled_sampling_unlabeled_data)
            unlabeled_dataset = unlabeled_object_wise_dataset("", test_txt, 40, 'unlabeled',
                                                              image_tmpl="_{:03d}." + img_ext,

                                                              max_num_views=views_number,
                                                              transform=torchvision.transforms.Compose(
                                                                  [transforms.ToTensor(),
                                                                   transforms.Normalize(
                                                                       mean=[0.485, 0.456,
                                                                             0.406],
                                                                       std=[0.229, 0.224, 0.225])
                                                                   ]),
                                                              selected_ind_train=unlabeled_sampling_labeled_data,
                                                              unselected_ind_train=unlabeled_sampling_unlabeled_data
                                                              )


            labeled_data = DataLoader(labeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True, worker_init_fn=tool.seed_worker)

            unlabeled_data = DataLoader(unlabeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                        shuffle=False,
                                        pin_memory=True, worker_init_fn=tool.seed_worker)

        if opt.DATA_SET == 'MVP_N':
            unlabeled_sampling_labeled_data = train_dataset.selected_ind_train
            unlabeled_sampling_labeled_data = [[item] for sublist in unlabeled_sampling_labeled_data for item in
                                               sublist]
            unlabeled_sampling_unlabeled_data = train_dataset.unselected_ind_train
            unlabeled_sampling_unlabeled_data = [[item] for sublist in unlabeled_sampling_unlabeled_data for item in
                                                 sublist]

            unlabeled_dataset = Unlabeled_Dataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'unlabeled',
                                                  opt.MAX_NUM_VIEWS, unlabeled_sampling_labeled_data,
                                                  unlabeled_sampling_unlabeled_data)
            unlabeled_data = DataLoader(unlabeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                        shuffle=False,
                                        pin_memory=True, worker_init_fn=tool.seed_worker)

            labeled_dataset = Unlabeled_Dataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'labeled',
                                                opt.MAX_NUM_VIEWS, unlabeled_sampling_labeled_data,
                                                unlabeled_sampling_unlabeled_data)
            labeled_data = DataLoader(labeled_dataset, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS,
                                      shuffle=False,
                                      pin_memory=True, worker_init_fn=tool.seed_worker)

        print(opt.QUERIES_STRATEGY)
        if opt.QUERIES_STRATEGY == 'uncertainty':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.uncertainty_sampling_one_label_multi_Ob(opt,
                                                                                                                    engine,
                                                                                                                    train_dataset,
                                                                                                                    unlabeled_data,
                                                                                                                    labeled_dataset)
        # view based selection
        if opt.QUERIES_STRATEGY == 'dissimilarity_sampling':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.dissimilarity_sampling_object_wise(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_data,
                train_data
            )

        if opt.QUERIES_STRATEGY == 'patch_based_selection':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.patch_based_selection_DAN(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_data,
                train_data,
                unlabeled_sampling_labeled_data,
                unlabeled_sampling_unlabeled_data
            )

        if opt.QUERIES_STRATEGY == 'course_to_fine':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.coursetofine(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_data,
                train_data,
                unlabeled_sampling_labeled_data,
                unlabeled_sampling_unlabeled_data,
                model_stage2
            )

        if opt.QUERIES_STRATEGY == 'random':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.random_sampling_object_wise(opt,
                                                                                                               engine,
                                                                                                               train_dataset,
                                                                                                               unlabeled_data,
                                                                                                               labeled_dataset)

        if opt.QUERIES_STRATEGY == 'LfOSA':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.LfOSA(opt,
                                                                                                     engine,
                                                                                                     train_dataset,
                                                                                                     unlabeled_data,
                                                                                                     labeled_dataset)

        if opt.QUERIES_STRATEGY == 'BGADL':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.bayesian_generative_active_learning(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_dataset)

        if opt.QUERIES_STRATEGY == 'OpenMax':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.open_max(opt,
                                                                                                        engine,
                                                                                                        train_data,
                                                                                                        train_dataset,
                                                                                                        unlabeled_data,
                                                                                                        )

        if opt.QUERIES_STRATEGY == 'Core_set':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.core_set(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_dataset)

        if opt.QUERIES_STRATEGY == 'BADGE_sampling':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.badge_sampling(
                opt,
                engine,
                train_dataset,
                labeled_data,
                unlabeled_data,
                train_data)

        if opt.QUERIES_STRATEGY == 'certainty':
            selected_ind_train_after_sampling, unselected_ind_train__after_sampling = sampling.certainty_sampling(
                opt,
                engine,
                train_dataset,
                unlabeled_data,
                labeled_dataset)

        print(len(selected_ind_train_after_sampling[0]))
        print(len(unselected_ind_train__after_sampling[0]))

        if opt.DATA_SET == 'M40v2':
            train_txt = 'v2_trainmodel40.txt'
            test_txt = 'v2_testmodel40.txt'
            img_ext = 'png'
            views_number = opt.MAX_NUM_VIEWS

            # if opt.DATA_SET == 'M40v2':
            opt.nb_classes = 40  # ModelNet40 class number

            train_dataset = object_wise_dataset("", train_txt, 40, 'train',

                                                image_tmpl="_{:03d}." + img_ext,

                                                max_num_views=views_number,
                                                view_number=opt.view_num,
                                                transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(
                                                                                  mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])
                                                                              ]),
                                                selected_ind_train=selected_ind_train_after_sampling,
                                                unselected_ind_train=unselected_ind_train__after_sampling
                                                )


            train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True, worker_init_fn=tool.seed_worker)

        if opt.DATA_SET == 'MVP_N':
            train_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'train', opt.MAX_NUM_VIEWS,
                                             use_train=True, selected_ind_train=selected_ind_train_after_sampling,
                                             unselected_ind_train=unselected_ind_train__after_sampling)
            train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=True,
                                    pin_memory=True, worker_init_fn=tool.seed_worker)

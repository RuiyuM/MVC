import os
import glob
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
class MultiViewDataset(Dataset):
    def __init__(self, classes, num_classes, data_root, mode, max_num_views, use_train=False, selected_ind_train=None, unselected_ind_train=None):
        super(MultiViewDataset, self).__init__()
 
        self.classes = classes
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.max_num_views = max_num_views
        self.use_train = use_train

        self.file_path = [[] for _ in range(len(self.classes))]
        self.none_train_path = []

        self.selected_ind_train = selected_ind_train
        self.unselected_ind_train = unselected_ind_train


        for class_name in self.classes:
            class_index = self.classes.index(class_name)
            if self.mode == 'train':
                mode_path = os.path.join(self.data_root, mode, class_name)
                for set_file in sorted(os.listdir(mode_path)):
                    files = sorted(glob.glob(os.path.join(mode_path, set_file, '*.png')))
                    for elements_ in files:

                        self.file_path[class_index].append(elements_)
            else:
                class_list = self.data_root + mode + '_list/' + mode + '_' + class_name + '_100.pkl'
                open_file = open(class_list, 'rb')
                list_path = pickle.load(open_file)
                open_file.close()
                self.none_train_path += list_path

        if self.use_train:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])

        if unselected_ind_train is None and selected_ind_train is None and self.mode == 'train':
            self.selected_ind_train, self.unselected_ind_train = self.filter_selected_unselected(self.file_path)

        self.data = []
        for index in range(self.__len__()):
            if self.mode == 'train':
                path = self.selected_ind_train
                split_setting = os.sep
            else:
                path = self.none_train_path
                split_setting = "/"

            class_name = path[index][0].split(split_setting)[-3]
            class_id = self.classes.index(class_name)
            label = torch.zeros(self.num_classes)
            label[class_id] = 1.0
            images = []
            marks = torch.zeros(self.max_num_views)

            for i in range(0, len(path[index])):
                image_name = (path[index][i].strip().split(os.sep)[-1]).strip().split('.')[0]
                marks[i] = int(image_name.strip().split('_')[-1])
                if self.mode == 'train':
                    image = Image.open(path[index][i]).convert('RGB')
                else:
                    image = Image.open(os.path.join(self.data_root, self.mode, path[index][i])).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                images.append(image)

            for i in range(0, self.max_num_views - len(path[index])):
                images.append(torch.zeros_like(images[0]))

            self.data.append((label, torch.stack(images), len(path[index]), marks))
    def filter_selected_unselected(self, file_path):
        selected_ind_train = [[] for _ in range(len(file_path))]
        unselected_ind_train = [[] for _ in range(len(file_path))]

        for i, class_files in enumerate(file_path):
            # Shuffle the list in-place
            random.shuffle(class_files)

            # Select one element for labeled_ind_train
            selected_ind_train[i].append(class_files[0])

            # The rest go to unlabeled_ind_train
            unselected_ind_train[i].extend(class_files[1:])

        return selected_ind_train, unselected_ind_train


    def __len__(self):
        if self.mode == 'train':
            return len(self.selected_ind_train)
        else:
            return len(self.none_train_path)

    # def __getitem__(self, index):
    #     if self.mode == 'train':
    #         path = self.selected_ind_train
    #         split_setting = os.sep
    #     else:
    #         path = self.none_train_path
    #         split_setting = "/"
    #     # print(path[index][0])
    #     class_name = path[index][0].split(split_setting)[-3]
    #     class_id = self.classes.index(class_name)
    #     label = torch.zeros(self.num_classes)
    #     label[class_id] = 1.0
    #     images = []
    #     marks = torch.zeros(self.max_num_views)
    #
    #     for i in range(0, len(path[index])):
    #         image_name = (path[index][i].strip().split(os.sep)[-1]).strip().split('.')[0]
    #         marks[i] = int(image_name.strip().split('_')[-1])
    #         if self.mode == 'train':
    #             image = Image.open(path[index][i]).convert('RGB')
    #         else:
    #             image = Image.open(os.path.join(self.data_root, self.mode, path[index][i])).convert('RGB')
    #
    #         if self.transform:
    #             image = self.transform(image)
    #
    #         images.append(image)
    #
    #     for i in range(0, self.max_num_views - len(path[index])):
    #         images.append(torch.zeros_like(images[0]))
    #
    #     return label, torch.stack(images), len(path[index]), marks

    def __getitem__(self, index):
        return self.data[index]

import os
import glob
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random


class Unlabeled_Dataset(Dataset):
    def __init__(self, classes, num_classes, data_root, mode, max_num_views, selected_ind_train=None,
                 unselected_ind_train=None, transform=None):
        super(Unlabeled_Dataset, self).__init__()

        self.classes = classes
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.max_num_views = max_num_views

        self.selected_ind_train = selected_ind_train
        self.unselected_ind_train = unselected_ind_train

        self.file_path = [[] for _ in range(len(self.classes))]

        for class_name in self.classes:
            class_index = self.classes.index(class_name)
            # if self.mode == 'train':
            mode_path = os.path.join(self.data_root, 'train', class_name)
            for set_file in sorted(os.listdir(mode_path)):
                files = sorted(glob.glob(os.path.join(mode_path, set_file, '*.png')))
                for elements_ in files:
                    self.file_path[class_index].append(elements_)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if unselected_ind_train is None and selected_ind_train is None:
            self.selected_ind_train, self.unselected_ind_train = self.filter_selected_unselected(self.file_path)


    def filter_selected_unselected(self, file_path):
        selected_ind_train = [[] for _ in range(len(file_path))]
        unselected_ind_train = []

        for i, class_files in enumerate(file_path):
            # Shuffle the list in-place
            random.shuffle(class_files)

            # Select one element for labeled_ind_train
            selected_ind_train[i].append(class_files[0])

            # The rest go to unlabeled_ind_train
            for file in class_files[1:]:
                unselected_ind_train.append([file])

        return selected_ind_train, unselected_ind_train

    def __len__(self):
        if self.mode == "labeled":
            return len(self.selected_ind_train)
        else:
            return len(self.unselected_ind_train)


    def __getitem__(self, index):
        if self.mode == "labeled":
            path = self.selected_ind_train
            end_point_setting = len(path[index])
        else:
            path = self.unselected_ind_train
            end_point_setting = len(path[index])
        split_setting = os.sep
        # print(path[index][0])
        train_path = path[index][0]
        class_name = path[index][0].split(split_setting)[-3]
        class_id = self.classes.index(class_name)
        label = torch.zeros(self.num_classes)
        label[class_id] = 1.0
        images = []
        marks = torch.zeros(self.max_num_views)




        for i in range(0, end_point_setting):
            image_name = (path[index][i].strip().split(os.sep)[-1]).strip().split('.')[0]
            marks[i] = int(image_name.strip().split('_')[-1])

            image = Image.open(path[index][i]).convert('RGB')


            if self.transform:
                image = self.transform(image)

            images.append(image)

        # for i in range(0, self.max_num_views - end_point_setting):
        #     images.append(torch.zeros_like(images[0]))
        # if self.mode == "labeled":
        #     return label, torch.stack(images), len(path[index]), marks
        # else:
        #     return label, torch.stack(images), len(path[index]), marks, train_path
        if self.mode == "labeled":
            return label, images, len(path[index]), marks
        else:
            return label, images, len(path[index]), marks, train_path


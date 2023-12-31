import torch.utils.data as data

from PIL import Image
import random
import torch
import os
import os.path
import numpy as np
from numpy.random import randint


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


# 这个数据集是一个object作为一个class
class object_wise_dataset(data.Dataset):
    def __init__(self, root_path, list_file, num_classes, mode,
                 image_tmpl='_{:03}.jpg', max_num_views=12, view_number=1, num_validation=2, validation_mode=False, transform=None,
                 selected_ind_train=None, unselected_ind_train=None
                 ):
        self.classes = list(range(num_classes))
        self.mode = mode
        self.num_classes = num_classes
        self.root_path = root_path
        self.list_file = list_file
        self.image_tmpl = image_tmpl
        self.transform = transform
        # this view_number is the dataset default number of view for each object
        self.view_number = view_number
        self.max_num_views = max_num_views
        # self.count = 0
        self._parse_list()
        self.file_path = [[] for _ in range(len(self.classes))]
        self.none_train_path = [[] for _ in range(len(self.classes))]
        self.selected_ind_train = selected_ind_train
        self.unselected_ind_train = unselected_ind_train
        self.num_validation = num_validation
        # validation mode is True means that we are validating each images in a single image style.
        self.validation_mode = validation_mode

        # if self.mode == 'train':
        for element in self.video_list:
            self.file_path[element.label].append(element.path)
        # if self.mode == 'valid':
        #     for element in self.video_list:
        #         self.none_train_path[element.label].append(element.path)

        if unselected_ind_train is None and selected_ind_train is None:
            self.selected_ind_train, self.unselected_ind_train = self.filter_selected_unselected(self.file_path)

        self.data = []
        # print(len(self.selected_ind_train) * len(self.selected_ind_train[0][0]))
        if self.mode == 'train':
            for current_class in range(len(self.selected_ind_train)):
                for current_object in self.selected_ind_train[current_class][0]:
                    label = torch.zeros(self.num_classes)
                    label[current_class] = 1.0
                    image_paths = [str(img_path[0]) for img_path in current_object]
                    self.data.append(
                        (label, image_paths, len(self.selected_ind_train[current_class]), current_object[-1][1]))
        if self.mode == 'valid' and self.validation_mode == True:
            for record in self.video_list:
                current_class = record.label
                view_indices = list(range(1, 20 + 1))
                label = torch.zeros(self.num_classes)
                label[current_class] = 1.0
                marks = torch.zeros(int(1))
                for idx in view_indices:
                # (current_class + self.image_tmpl.format(selected_number)
                    image_paths = record.path + self.image_tmpl.format(idx)


                    self.data.append(
                        (label, [image_paths], 1, marks))


    def filter_selected_unselected(self, file_path):
        selected_ind_train = [[] for _ in range(len(file_path))]
        unselected_ind_train = [[] for _ in range(len(file_path))]

        for i, class_files in enumerate(file_path):
            # Shuffle the list in-place
            random.shuffle(class_files)
            # select top 50% of the object in that class
            class_files = class_files[:len(class_files) // 20]
            class_differentiater_selected = [[] for _ in range(len(class_files))]
            class_differentiater_unselected = [[] for _ in range(len(class_files))]
            for idx in range(len(class_files)):
                current_class = class_files[idx]
                view_list = list(range(21))
                view_list = view_list[1:]
                # print(view_list[0: 10])
                selected_number = random.choice(view_list)

                # Step 4: Create a new list containing the numbers from class_list except the selected number
                new_list = [num for num in view_list if num != selected_number]

                # Select max_num_views element for labeled_ind_train
                class_differentiater_selected[idx].append((current_class + self.image_tmpl.format(selected_number), idx))

                # The rest go to unlabeled_ind_train
                for view_number in new_list:
                    class_differentiater_unselected[idx].append((current_class + self.image_tmpl.format(view_number), idx))
            selected_ind_train[i].append(class_differentiater_selected)
            unselected_ind_train[i].append(class_differentiater_unselected)


        return selected_ind_train, unselected_ind_train

    def _load_image(self, directory, idx):
        image_path = directory + self.image_tmpl.format(idx)
        return [Image.open(directory + self.image_tmpl.format(idx)).convert('RGB')], image_path

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        if self.mode == 'train':
            label, image_paths, num_selected, obj_info = self.data[index]
            images = []
            for img_path in image_paths:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return label, torch.stack(images), num_selected, obj_info


        elif self.mode == 'valid' and self.validation_mode == False:
            # change the following number (currently is 20) can change the
            # number of views been used at validation stage
            record = self.video_list[index]
            current_class = record.label
            view_indices = list(range(1, 20 + 1))
            # random.shuffle(view_indices)
            view_indices = view_indices[:int(self.num_validation)]
            images = []
            label = torch.zeros(self.num_classes)
            label[current_class] = 1.0
            marks = torch.zeros(int(self.num_validation))
            for idx in view_indices:

                image = Image.open(record.path + self.image_tmpl.format(idx)).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)

            return label, torch.stack(images), int(self.num_validation), marks

        elif self.mode == 'valid' and self.validation_mode == True:
            label, image_paths, num_selected, obj_info = self.data[index]
            images = []
            for img_path in image_paths:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return label, torch.stack(images), num_selected, obj_info


    def get(self, record, indices, index):
        images = list()
        for view_idx in indices:
            seg_imgs = self._load_image(record.path, view_idx)
            images.extend(seg_imgs)
        process_data1 = self.transform(images)

        return process_data1, record.label

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'valid'and self.validation_mode == False:
            return len(self.video_list)
        elif self.mode == 'valid' and self.validation_mode == True:
            return len(self.data)


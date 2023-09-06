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



class MVDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_classes, mode,
                 image_tmpl='_{:03}.jpg', max_num_views=12, view_number=1, transform=None,
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


        if self.mode == 'train':
            for element in self.video_list:
                self.file_path[element.label].append(element.path)
        if self.mode == 'valid':
            for element in self.video_list:
                self.none_train_path[element.label].append(element.path)

        if unselected_ind_train is None and selected_ind_train is None:
            self.selected_ind_train, self.unselected_ind_train = self.filter_selected_unselected(self.file_path)

        self.data = []
        print(len(self.selected_ind_train) * len(self.selected_ind_train[0][0]))
        if self.mode == 'train':

            marks = torch.zeros(self.max_num_views)

            for current_class in range(len(self.selected_ind_train)):
                # label = torch.full((self.max_num_views,), current_class, dtype=torch.int)
                label = torch.zeros(self.num_classes)
                label[current_class] = 1.0
                number_of_image_tuple = int(len(self.selected_ind_train[current_class][0]))
                for idx in range(number_of_image_tuple):
                    images = []
                    for i in range(1, 1 + self.max_num_views):
                        # print(self.selected_ind_train[current_class][0][idx] + self.image_tmpl.format(i))
                        image = Image.open(self.selected_ind_train[current_class][0][idx] + self.image_tmpl.format(i)).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        images.append(image)
                    self.data.append((label, torch.stack(images), int(self.max_num_views), marks))

        if self.mode == 'sampling':
            images = []
            marks = torch.zeros(self.view_number)

            for current_class in range(len(self.unselected_ind_train)):
                label = torch.zeros(self.num_classes)
                label[current_class] = 1.0
                # label = torch.full((self.view_number,), current_class, dtype=torch.int)
                number_of_image_tuple = int(len(self.unselected_ind_train[current_class][0]))
                for idx in range(number_of_image_tuple):
                    for i in range(1, 1 + self.view_number):
                        # print(self.unselected_ind_train[current_class][0][idx] + self.image_tmpl.format(i))
                        image = [Image.open(
                            self.unselected_ind_train[current_class][0][idx] + self.image_tmpl.format(i)).convert('RGB')]
                        if self.transform:
                            image = self.transform(image)
                        images.append(image)
                    self.data.append((label, torch.stack(images), int(self.view_number), marks))

        if self.mode == 'valid':
            images = []
            marks = torch.zeros(self.view_number)

            for current_class in range(len(self.none_train_path)):
                label = torch.full((self.view_number,), current_class, dtype=torch.int)
                number_of_image_tuple = int(len(self.none_train_path[current_class]))
                for idx in range(number_of_image_tuple):
                    for i in range(1, 1 + self.view_number):
                        # print(self.none_train_path[current_class][0][idx] + self.image_tmpl.format(i))
                        image = [Image.open(self.none_train_path[current_class][idx] + self.image_tmpl.format(i)).convert(
                            'RGB')]
                        if self.transform:
                            image = self.transform(image)
                        images.append(image)
                    self.data.append((label, torch.stack(images), self.view_number, marks))










    def filter_selected_unselected(self, file_path):
        selected_ind_train = [[] for _ in range(len(file_path))]
        unselected_ind_train = [[] for _ in range(len(file_path))]

        for i, class_files in enumerate(file_path):
            # Shuffle the list in-place
            random.shuffle(class_files)

            # Select max_num_views element for labeled_ind_train
            selected_ind_train[i].append(class_files[:self.view_number])

            # The rest go to unlabeled_ind_train
            unselected_ind_train[i].extend(class_files[self.view_number:])

        return selected_ind_train, unselected_ind_train

    def _load_image(self, directory, idx):
        image_path = directory + self.image_tmpl.format(idx)
        return [Image.open(directory + self.image_tmpl.format(idx)).convert('RGB')], image_path

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        return self.data[index]

    # def __getitem__(self, index):
    #     record = self.video_list[index]
    #     view_indices = np.linspace(1, self.total_view, self.view_number, dtype=int)
    #     self.count += 1
    #     print(self.count)
    #     return self.get(record, view_indices, index)

    # def __getitem__(self, index):
    #     # Calculate which video record and which frame within that record
    #     video_index = index // 20  # there are 20 images per record
    #     frame_index = index % 20 + 1  # +1 because indices start from 1
    #
    #     record = self.video_list[video_index]
    #     image, image_path = self._load_image(record.path, frame_index)
    #     if self.transform:
    #         image = self.transform(image)
    #     self.count += 1
    #     print(self.count)
    #     return image, record.label, image_path

    def get(self, record, indices, index):
        images = list()
        for view_idx in indices:
            seg_imgs = self._load_image(record.path, view_idx)
            images.extend(seg_imgs)
        process_data1 = self.transform(images)

        return process_data1, record.label

    def __len__(self):
        if self.mode == 'train':
            return len(self.selected_ind_train) * len(self.selected_ind_train[0][0])
        if self.mode == 'valid':
            return len(self.video_list)
        elif self.mode == 'sampling':
            return len(self.video_list) - len(self.selected_ind_train) * len(self.selected_ind_train[0][0])
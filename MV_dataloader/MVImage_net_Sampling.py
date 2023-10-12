import torch.utils.data as data
from PIL import Image
import os
import torch
import random


class SamplingDataset(data.Dataset):
    def __init__(self, root_path, num_classes, mode, transform=None, selected_ind_train=None, unselected_ind_train=None):
        self.root_path = root_path
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes
        self.data = []
        self.selected_ind_train = selected_ind_train
        self.unselected_ind_train = unselected_ind_train

        self._load_data()

        if self.unselected_ind_train is None and self.selected_ind_train is None:
            self.selected_ind_train, self.unselected_ind_train = self.filter_selected_unselected(self.file_path)

        if self.mode == 'labeled':

            for current_class, class_data in enumerate(self.selected_ind_train):
                for current_object in class_data[0]:
                    label = torch.zeros(len(self.selected_ind_train))
                    label[current_class] = 1.0
                    for img_path in current_object:
                        image_paths = str(img_path[0])
                        self.data.append(
                            (label, [image_paths], len(self.selected_ind_train[current_class]), current_object[-1][1]))

        if self.mode == 'unlabeled':
            for current_class, class_data in enumerate(self.unselected_ind_train):
                for current_object in class_data[0]:
                    label = torch.zeros(len(self.unselected_ind_train))
                    label[current_class] = 1.0
                    for img_path in current_object:
                        image_paths = str(img_path[0])
                        self.data.append(
                            (label, [image_paths], len(self.unselected_ind_train[current_class]), current_object[-1][1]))

    def _load_data(self):
        self.class_dirs = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
        self.file_path = {}
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.root_path, class_dir)
            model_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            for model_dir in model_dirs:
                image_dir = os.path.join(class_path, model_dir, 'images')
                image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

                # Check if class_dir is not in the dictionary, if not, add it with an empty list
                if class_dir not in self.file_path:
                    self.file_path[class_dir] = []

                # Append image files to the list associated with class_dir
                self.file_path[class_dir].append(image_files)
        sorted_file_path = sorted(self.file_path.items(), key=lambda item: len(item[1]), reverse=True)

        # Keep only the top 40 longest lists
        self.file_path = dict(sorted_file_path[:40])

    def filter_selected_unselected(self, file_path):
        selected_ind_train = [[] for _ in range(len(file_path))]
        unselected_ind_train = [[] for _ in range(len(file_path))]

        for i, (class_name, class_files) in enumerate(file_path.items()):
            random.shuffle(class_files)
            # flat_class_files = class_files[:len(class_files) // 20]
            class_differentiater_selected = [[] for _ in range(len(class_files))]
            class_differentiater_unselected = [[] for _ in range(len(class_files))]
            for idx, current_class in enumerate(class_files):
                selected_view = random.choice(current_class)

                # Step 3: Create a new list that contains all other elements
                remaining_view = [student for student in current_class if student != selected_view]
                class_differentiater_selected[idx].append((selected_view, idx))
                for element in remaining_view:
                    class_differentiater_unselected[idx].append((element, idx))
            selected_ind_train[i].append(class_differentiater_selected)
            unselected_ind_train[i].append(class_differentiater_unselected)

        return selected_ind_train, unselected_ind_train

    def __getitem__(self, index):
        if self.mode == 'labeled':
            label, image_paths, num_selected, obj_info = self.data[index]
            images = []
            for img_path in image_paths:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return label, torch.stack(images), num_selected, obj_info

        else:
            label, image_paths, num_selected, obj_info = self.data[index]
            images = []
            for img_path in image_paths:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return label, torch.stack(images), num_selected, image_paths , obj_info



    def __len__(self):
        return len(self.data)
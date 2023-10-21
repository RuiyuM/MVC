import numpy as np
from torch.autograd import Variable
import torch
import random
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
import torch.nn.functional as F
from copy import deepcopy
import pdb
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import timm
import tome
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from unlabeled_Sampling_Dataset import Unlabeled_Dataset
from torch.utils.data import DataLoader
import utils as tool
from multiprocessing import Pool
import time
import copy
from tome.utils import parse_r
from timm.models.layers.patch_embed import PatchEmbed
import torch.nn as nn
import torchvision.models as models
from main_multi_view import generate_sampling_dataset





def patch_based_selection_DAN(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data,

                          ):
    engine.model.eval()
    # the label_K_dict is a variable that store the K value for selected object's view
    label_metric_dict = {}
    with torch.no_grad():

        for index, (label, image, num_views, object_class) in enumerate(labeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)


            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar

                new_metric_list = k_metrics[i]
                if true_label not in label_metric_dict:
                    label_metric_dict[true_label] = []
                    # Add new_metric_list to dictionary
                label_metric_dict[true_label].append(new_metric_list)
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()

    with torch.no_grad():
        # training_K_selected_dict is a variable which store the K for the rest unselected objects' view
        training_metric_label_dict = {}

        # First pass: collect features and paths
        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)
            true_labels = torch.max(targets, 1)[1]  # This is now a tensor of labels for the batch


            # Loop over the batch
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = k_metrics[i]
                path = train_path[i]  # Get the train path for this image
                if true_label not in training_metric_label_dict:
                    training_metric_label_dict[true_label] = []
                # Add new_metric_list and train_path to dictionary
                training_metric_label_dict[true_label].append([new_metric_list, path])
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()


        selected_path = calculate_similarity_bipartite(label_metric_dict, training_metric_label_dict)

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(len(selected_path)):
        # Get the 10 least similar paths for this class
        # Assuming the paths are sorted in ascending order of similarity
        least_similar_paths = selected_path[class_index]

        for least_similar_path in least_similar_paths:
            # Append the least_similar_path to the corresponding sublist in old_index_train
            old_index_train[class_index].append(least_similar_path)

            # Remove the least_similar_path from the sublist in old_index_not_train at class_index
            # Assuming old_index_not_train is a list of lists where each sublist corresponds to a class and contains the paths of the images for that class
            if least_similar_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(least_similar_path)

    return old_index_train, old_index_not_train

def reverse_patch_based_selection_DAN(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data,

                          ):
    engine.model.eval()
    # the label_K_dict is a variable that store the K value for selected object's view
    label_metric_dict = {}
    with torch.no_grad():

        for index, (label, image, num_views, object_class) in enumerate(labeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)


            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar

                new_metric_list = k_metrics[i]
                if true_label not in label_metric_dict:
                    label_metric_dict[true_label] = []
                    # Add new_metric_list to dictionary
                label_metric_dict[true_label].append(new_metric_list)
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()

    with torch.no_grad():
        # training_K_selected_dict is a variable which store the K for the rest unselected objects' view
        training_metric_label_dict = {}

        # First pass: collect features and paths
        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)
            true_labels = torch.max(targets, 1)[1]  # This is now a tensor of labels for the batch


            # Loop over the batch
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = k_metrics[i]
                path = train_path[i]  # Get the train path for this image
                if true_label not in training_metric_label_dict:
                    training_metric_label_dict[true_label] = []
                # Add new_metric_list and train_path to dictionary
                training_metric_label_dict[true_label].append([new_metric_list, path])
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()


        selected_path = reverse_calculate_similarity_bipartite(label_metric_dict, training_metric_label_dict)

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(len(selected_path)):
        # Get the 10 least similar paths for this class
        # Assuming the paths are sorted in ascending order of similarity
        least_similar_paths = selected_path[class_index]

        for least_similar_path in least_similar_paths:
            # Append the least_similar_path to the corresponding sublist in old_index_train
            old_index_train[class_index].append(least_similar_path)

            # Remove the least_similar_path from the sublist in old_index_not_train at class_index
            # Assuming old_index_not_train is a list of lists where each sublist corresponds to a class and contains the paths of the images for that class
            if least_similar_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(least_similar_path)

    return old_index_train, old_index_not_train

def calculate_similarity_bipartite(label_metric_dict, training_metric_label_dict):
    selected_paths = [[] for _ in range(len(label_metric_dict))]

    for true_label, label_metrics_list in label_metric_dict.items():
        if true_label not in training_metric_label_dict:
            continue

        for training_metrics, path in training_metric_label_dict[true_label]:
            current_min_cost = float('inf')

            # Loop over each set of metrics for this class in the labeled data
            for label_metrics in label_metrics_list:
                label_metrics = F.normalize(label_metrics, dim=0)
                training_metrics = F.normalize(training_metrics, dim=0)
                cost_matrix = torch.mm(label_metrics.t(), training_metrics)  # 196*196
                # cost_matrix = torch.mm(label_metrics, training_metrics.t())
                cost_matrix = -(cost_matrix + 1)
                cost_matrix_np = cost_matrix.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                total_cost = cost_matrix_np[row_ind, col_ind].sum()

                if total_cost < current_min_cost:
                    current_min_cost = total_cost

            # Append the minimum cost and associated path for this unlabeled sample
            selected_paths[true_label].append([current_min_cost, path])

    new_list = []

    # Loop through the list of classes
    for paths in selected_paths:
        # Sort the list in descending order (highest first)
        sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)

        # Select the top 5 biggest and store their paths
        top_5_paths_for_class = sorted_paths[:2]  # Get the first five items from the sorted list

        # Create a new list to store the paths of the top 5 biggest
        paths_list = []
        for item in top_5_paths_for_class:
            # Extract the paths only
            path = item[1]
            paths_list.append(path)

        # Append the paths to the new_list
        new_list.append(paths_list)


    return new_list

def reverse_calculate_similarity_bipartite(label_metric_dict, training_metric_label_dict):
    selected_paths = [[] for _ in range(len(label_metric_dict))]

    for true_label, label_metrics_list in label_metric_dict.items():
        if true_label not in training_metric_label_dict:
            continue

        for training_metrics, path in training_metric_label_dict[true_label]:
            current_min_cost = float('inf')

            # Loop over each set of metrics for this class in the labeled data
            for label_metrics in label_metrics_list:
                label_metrics = F.normalize(label_metrics, dim=0)
                training_metrics = F.normalize(training_metrics, dim=0)
                cost_matrix = torch.mm(label_metrics.t(), training_metrics)  # 196*196
                # cost_matrix = torch.mm(label_metrics, training_metrics.t())
                cost_matrix = -(cost_matrix + 1)
                cost_matrix_np = cost_matrix.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                total_cost = cost_matrix_np[row_ind, col_ind].sum()

                if total_cost < current_min_cost:
                    current_min_cost = total_cost

            # Append the minimum cost and associated path for this unlabeled sample
            selected_paths[true_label].append([current_min_cost, path])

    new_list = []

    # Loop through the list of classes
    for paths in selected_paths:
        # Sort the list in descending order (highest first)
        sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)

        # Select the top 5 biggest and store their paths
        top_5_paths_for_class = sorted_paths[:2]  # Get the first five items from the sorted list

        # Create a new list to store the paths of the top 5 biggest
        paths_list = []
        for item in top_5_paths_for_class:
            # Extract the paths only
            path = item[1]
            paths_list.append(path)

        # Append the paths to the new_list
        new_list.append(paths_list)


    return new_list

def patch_based_selection_DAN_object_wise(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data,

                          ):
    engine.model.eval()
    # the label_K_dict is a variable that store the K value for selected object's view
    label_K_dict = [{} for _ in range(opt.nb_classes)]
    with torch.no_grad():

        for index, (label, image, num_views, object_class) in enumerate(labeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)


            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar

                new_metric_list = k_metrics[i]
                if object_class[i].item() not in label_K_dict[true_label]:
                    label_K_dict[true_label][object_class[i].item()] = []
                # Add new_metric_list to dictionary
                label_K_dict[true_label][object_class[i].item()].append(new_metric_list)
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()

    with torch.no_grad():
        # training_K_selected_dict is a variable which store the K for the rest unselected objects' view
        training_K_selected_dict = [{} for _ in range(opt.nb_classes)]

        # First pass: collect features and paths
        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            # outputs is prediction, metrics is K, features is features before linear layer
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)
            true_labels = torch.max(targets, 1)[1]  # This is now a tensor of labels for the batch


            # Loop over the batch
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = k_metrics[i]
                path = train_path[i]  # Get the train path for this image
                if object_class[i].item() not in training_K_selected_dict[true_label]:
                    training_K_selected_dict[true_label][object_class[i].item()] = []
                # Add new_metric_list and train_path to dictionary
                training_K_selected_dict[true_label][object_class[i].item()].append([new_metric_list, path])
            del inputs, targets, outputs, features
            torch.cuda.empty_cache()


        selected_path = calculate_similarity_bipartite_object_wise(label_K_dict, training_K_selected_dict)

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(len(selected_path)):
        # Get the 10 least similar paths for this class
        # Assuming the paths are sorted in ascending order of similarity
        least_similar_paths = selected_path[class_index]

        # for least_similar_path in least_similar_paths:
            # Append the least_similar_path to the corresponding sublist in old_index_train
        for idx in range(len(least_similar_paths[0])):
            old_index_train[class_index][0][idx].append(least_similar_paths[0][idx][0])
            # print(len(old_index_not_train[class_index][0][idx]))
            for jdx in range(len(old_index_not_train[class_index][0][idx])):
                x1 = least_similar_paths[0][idx][0][0]
                x2 = old_index_not_train[class_index][0][idx][jdx][0]
                if x1 == x2:
                    del old_index_not_train[class_index][0][idx][jdx]
                    break


            # Remove the least_similar_path from the sublist in old_index_not_train at class_index
            # Assuming old_index_not_train is a list of lists where each sublist corresponds to a class and contains the paths of the images for that class


    return old_index_train, old_index_not_train


# def change_patch_size(transformer_class):
#     class Change_patch_size_class(transformer_class):
#         """
#         Modifications:
#         - Initialize r, token size, and token sources.
#         - For MAE: make global average pooling proportional to token size
#         """
#
#         def forward(self, *args, **kwdargs) -> torch.Tensor:
#             self._tome_info["r"] = parse_r(len(self.blocks), self.r)
#             self._tome_info["size"] = None
#             self._tome_info["source"] = None
#             # last_element = F.avg_pool2d(args[-1], kernel_size=2, stride=2)
#             # modified_args = args[:-1] + (last_element,)
#
#             return super().forward(*args, **kwdargs)
#
#     return Change_patch_size_class

class ReducePatch(PatchEmbed):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        # mobilenet_v2.cuda()
        # features = nn.Sequential(*list(mobilenet_v2.features.children())[:7])
        resnet18 = models.resnet18(pretrained=True).cuda()
        resnet18.eval()
        # Use layers up to the third layer (this will give 1/4 downsample for 224x224 input)
        features = nn.Sequential(*list(resnet18.children())[:-4])


        x = features(x)
        channel_reducer = nn.Conv2d(x.size(1), 3, kernel_size=1).cuda()
        x = channel_reducer(x)

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def coursetofine(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data,
                          unlabeled_sampling_labeled_data,
                          unlabeled_sampling_unlabeled_data, model_stage2):
    engine.model.eval()
    label_metric_dict = [{} for _ in range(opt.nb_classes)]
    # for module in model_stage2.modules():
    #     if isinstance(module, PatchEmbed):
    #         module.__class__ = ReducePatch
    model_stage2.r = 20
    with torch.no_grad():

        for index, (label, image, num_views, object_class) in enumerate(labeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            # print(true_labels.shape)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, metrics, features = engine.model(B, V, num_views, inputs)
            # batch_size, token_dimension = features.shape
            # metrics = engine.model.k_value.reshape(len(true_labels), 1, token_dimension)

            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = metrics[i]
                token_number, token_dimension = new_metric_list.shape
                new_metric_list = new_metric_list.reshape(token_number * token_dimension)
                # Check if the label exists in the dictionary
                # print(object_class[i].item())
                if object_class[i].item() not in label_metric_dict[true_label]:
                    label_metric_dict[true_label][object_class[i].item()] = []
                # Add new_metric_list to dictionary
                label_metric_dict[true_label][object_class[i].item()].append(new_metric_list)

    with torch.no_grad():
        training_metric_label_dict = [{} for _ in range(opt.nb_classes)]

        # First pass: collect features and paths
        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, metrics, features = engine.model(B, V, num_views, inputs)
            true_labels = torch.max(targets, 1)[1]  # This is now a tensor of labels for the batch

            # batch_size, token_dimension = features.shape
            # metrics = engine.model.k_value.reshape(len(true_labels), 1, token_dimension)

            # Loop over the batch
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = metrics[i]
                token_number, token_dimension = new_metric_list.shape
                new_metric_list = new_metric_list.reshape(token_number * token_dimension)
                path = train_path[i]  # Get the train path for this image
                if object_class[i].item() not in training_metric_label_dict[true_label]:
                    training_metric_label_dict[true_label][object_class[i].item()] = []
                # Add new_metric_list and train_path to dictionary
                training_metric_label_dict[true_label][object_class[i].item()].append([new_metric_list, path])

        old_index_not_train = get_dissimilar_paths_list(training_metric_label_dict, label_metric_dict)
        old_index_train = train_dataset.selected_ind_train

        labeled_data, unlabeled_data = generate_sampling_dataset(old_index_train, old_index_not_train, opt)
        engine.model.r = 0
        old_index_train, old_index_not_train = patch_based_selection_DAN(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data,
                                      unlabeled_sampling_labeled_data,
                                      unlabeled_sampling_unlabeled_data)

    return old_index_train, old_index_not_train


def compute_distance(matrix1, matrix2):
    # Using Euclidean distance as an example
    return torch.norm(matrix1 - matrix2)


# def get_dissimilar_paths_list(training_metric_label_dict, label_metric_dict):
#     # Initialize the new list with empty dictionaries
#     new_list = [{} for _ in range(len(training_metric_label_dict))]
#
#     # Iterate over each coarse label
#     for coarse_label in range(len(training_metric_label_dict)):
#         # Iterate over each fine label
#         for fine_label in training_metric_label_dict[coarse_label].keys():
#             distances = []
#             # For each matrix in training_metric_label_dict, compute its distance to the matrix in label_metric_dict
#             for metric_list, path in training_metric_label_dict[coarse_label][fine_label]:
#                 label_metric = label_metric_dict[coarse_label][fine_label][0]
#                 distance = compute_distance(metric_list, label_metric)
#                 distances.append((distance, metric_list, path))
#
#             # Sort by distance in descending order and select the top 5
#             sorted_distances = sorted(distances, key=lambda x: x[0], reverse=True)[:5]
#
#             # Store the top 5 most dissimilar matrices in the new list
#             new_list[coarse_label][fine_label] = [(metric_list, path) for _, metric_list, path in sorted_distances]
#
#     return new_list

def get_dissimilar_paths_list(training_metric_label_dict, label_metric_dict):
    # Initialize the new list with empty lists
    new_list = [[[[] for _ in training_metric_label_dict[coarse_label]]] for coarse_label in range(len(training_metric_label_dict))]

    # Iterate over each coarse label
    for coarse_label in range(len(training_metric_label_dict)):
        # Iterate over each fine label
        for fine_label_idx, fine_label in enumerate(training_metric_label_dict[coarse_label].keys()):
            distances = []
            # For each matrix in training_metric_label_dict, compute its distance to the matrix in label_metric_dict
            for metric_list, path in training_metric_label_dict[coarse_label][fine_label]:
                label_metric = label_metric_dict[coarse_label][fine_label][0]
                distance = compute_distance(metric_list, label_metric)
                distances.append((distance, path))

            # Sort by distance in descending order and select the top 5
            sorted_distances = sorted(distances, key=lambda x: x[0], reverse=True)[:5]

            # Store the top 5 most dissimilar paths in the new list
            for _, path in sorted_distances:
                new_list[coarse_label][0][fine_label_idx].append((path, fine_label))

    return new_list




def calculate_distance_DAN(args):
    label_metrics, training_metrics, true_label, path = args
    label_metrics = label_metrics.reshape(-1)
    training_metrics = training_metrics.reshape(-1)
    normalized_label_metrics = F.normalize(label_metrics, p=2, dim=0)
    normalized_training_metrics = F.normalize(training_metrics, p=2, dim=0)

    vec1 = normalized_label_metrics.view(len(normalized_label_metrics), 1)
    vec2 = normalized_training_metrics.view(1, len(normalized_training_metrics))
    cost_matrix = torch.matmul(vec1, vec2)
    cost_matrix_np = cost_matrix.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    total_cost = cost_matrix_np[row_ind, col_ind].sum()

    return true_label, total_cost, path


def calculate_distance(x, y):
    # Initialize variables to store the sum of minimum distances
    sum_min_distance_xy = 0
    sum_min_distance_yx = 0

    # Number of feature vectors in each set
    n_x = x.shape[0]
    n_y = y.shape[0]

    # Calculate the sum of minimum distances from x to y
    for i in range(n_x):
        min_distance = np.min(np.linalg.norm(y - x[i], axis=1))
        sum_min_distance_xy += min_distance

    # Calculate the sum of minimum distances from y to x
    for j in range(n_y):
        min_distance = np.min(np.linalg.norm(x - y[j], axis=1))
        sum_min_distance_yx += min_distance

    # Calculate the final distance
    distance = (sum_min_distance_xy / (2 * n_y)) + (sum_min_distance_yx / (2 * n_x))

    return distance


# def calculate_similarity_bipartite(label_metric_dict, training_metric_label_dict):
#     selected_paths = [[] for _ in range(len(label_metric_dict))]
#
#     for true_label, label_metrics_list in label_metric_dict.items():
#         if true_label not in training_metric_label_dict:
#             continue
#
#         for training_metrics, path in training_metric_label_dict[true_label]:
#             current_min_cost = float('inf')
#
#             # Loop over each set of metrics for this class in the labeled data
#             for label_metrics in label_metrics_list:
#                 # cost_matrix = torch.zeros((len(label_metrics), len(training_metrics)), device='cuda')
#
#                 cost_distance = calculate_distance(label_metrics, training_metrics)
#
#                 label_metrics = label_metrics.reshape(-1)
#                 training_metrics = training_metrics.reshape(-1)
#                 normalized_label_metrics = F.normalize(label_metrics, p=2, dim=0)
#
#                 # Element-wise L2 normalization for training_metrics
#                 normalized_training_metrics = F.normalize(training_metrics, p=2, dim=0)
#
#                 # Compute cosine similarity
#                 # Compute pairwise similarity matrix
#                 vec1 = normalized_label_metrics.view(len(normalized_label_metrics), 1)
#                 vec2 = normalized_training_metrics.view(1, len(normalized_training_metrics))
#
#                 # Step 2: Compute Pairwise Similarity
#                 cost_matrix = torch.matmul(vec1, vec2)
#
#                 # Convert the cost_matrix to a NumPy array for linear_sum_assignment
#                 cost_matrix_np = cost_matrix.cpu().numpy()
#
#                 # Apply linear_sum_assignment to find the optimal correspondence
#                 row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
#
#                 # Calculate the total cost for this correspondence
#                 total_cost = cost_matrix_np[row_ind, col_ind].sum()
#
#                 # Update the current_min_cost if this total_cost is smaller
#                 if total_cost < current_min_cost:
#                     current_min_cost = total_cost
#
#             # Append the minimum cost and associated path for this unlabeled sample
#             selected_paths[true_label].append([current_min_cost, path])
#
#     new_list = []
#     score_ = []
#     not_selected_list = []
#     not_selected_score = []
#     # Loop through the list of classes
#     for paths in selected_paths:
#         # Sort the list in descending order (highest first)
#         sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)
#
#         # Select the top 10 biggest and store their paths
#         selected_paths_for_class = sorted_paths[0]
#         unselected_paths_for_class = sorted_paths[-1]
#         score_.append(selected_paths_for_class[0])
#         # Extract the paths only
#         selected_paths_for_class = selected_paths_for_class[1]
#         not_selected_list.append(unselected_paths_for_class[1])
#         not_selected_score.append(unselected_paths_for_class[0])
#
#         new_list.append(selected_paths_for_class)
#
#     print(new_list)
#     print(score_)
#     print(not_selected_list)
#     print(not_selected_score)
#     return new_list

def calculate_similarity_bipartite_object_wise(label_metric_dicts, training_metric_label_dicts):
    # so the structures for label_metric_dicts and training_metric_label_dicts are
    # class -> object_class -> K; so we need to loop through them to extract the K values and perform
    # the calculation.
    # we first find the most similar view within object_class and find the most dissimilar in class wise
    start_time = time.time()
    selected_paths = [[] for _ in range(len(label_metric_dicts))]
    new_list = [[] for _ in range(len(label_metric_dicts))]
    for i, label_metric_dict in enumerate(label_metric_dicts):
        current_path = [[] for _ in range(len(label_metric_dict))]
        new_list[i].append(copy.deepcopy(current_path))
        for true_label, label_metrics_list in label_metric_dict.items():

            if true_label not in training_metric_label_dicts[i]:
                continue

            for training_metrics, path in training_metric_label_dicts[i][true_label]:
                current_min_cost = float('inf')

                for label_metrics in label_metrics_list:
                    # cost_distance = calculate_distance(label_metrics, training_metrics)
                    # 196 * 768 @ 768 * 196 = 196 * 196

                    # training_metrics = training_metrics.t()
                    label_metrics = F.normalize(label_metrics, dim=0)
                    training_metrics = F.normalize(training_metrics, dim=0)
                    cost_matrix = torch.mm(label_metrics.t(), training_metrics) # 196*196
                    # cost_matrix = torch.mm(label_metrics, training_metrics.t())
                    cost_matrix = -(cost_matrix + 1)
                    cost_matrix_np = cost_matrix.cpu().numpy()

                    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                    total_cost = cost_matrix_np[row_ind, col_ind].sum()

                    if total_cost < current_min_cost:
                        current_min_cost = total_cost
                current_path[true_label].append([current_min_cost, path])
                # selected_paths[true_label].append(path)

        selected_paths[i].append(current_path)

    end_time = time.time()
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total execution time: {end_time - start_time} seconds")
    # new_list = []

    # 在最相似的里面找出最不相似的那个
    for idx in range(len(selected_paths)):
        selected_path = selected_paths[idx]
        for jdx in range(len(selected_path[0])):
            paths = selected_path[0][jdx]
            sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)
            selected_paths_for_class = sorted_paths[0][1]
            new_list[idx][0][jdx].append((selected_paths_for_class, jdx))

    return new_list


# def calculate_similarity_bipartite(label_metric_dicts, training_metric_label_dicts):
#     selected_paths = [[] for _ in range(len(label_metric_dicts))]
#     new_list = [[] for _ in range(len(label_metric_dicts))]
#     start_time = time.time()
#     for i, label_metric_dict in enumerate(label_metric_dicts):
#         current_path = [[] for _ in range(len(label_metric_dict))]
#         new_list[i].append(current_path)
#
#         for true_label, label_metrics_list in label_metric_dict.items():
#             if true_label not in training_metric_label_dicts[i]:
#                 continue
#
#             args_list = [(label_metrics, training_metrics, true_label, path)
#                          for training_metrics, path in training_metric_label_dicts[i][true_label]
#                          for label_metrics in label_metrics_list]
#
#             with Pool() as pool:
#                 results = pool.map(calculate_distance_DAN, args_list)
#
#             current_min_cost = min(results, key=lambda x: x[1])[1]
#             current_path[true_label].append(min(results, key=lambda x: x[1])[1:])
#             # count += 1
#             # print(count)
#
#         selected_paths[i].append(current_path)
#     end_time = time.time()
#     print(f"Start time: {start_time}")
#     print(f"End time: {end_time}")
#     print(f"Total execution time: {end_time - start_time} seconds")
#     # new_list = []
#     score_ = []
#     not_selected_list = []
#     not_selected_score = []
#
#     for idx in range(len(selected_paths)):
#         selected_path = selected_paths[idx]
#         for jdx in range(len(selected_path)):
#             paths = selected_path[0][jdx]
#             sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)
#             selected_paths_for_class = sorted_paths[0]
#             unselected_paths_for_class = sorted_paths[-1]
#             score_.append(selected_paths_for_class[0])
#             selected_paths_for_class = selected_paths_for_class[1]
#             not_selected_list.append(unselected_paths_for_class[1])
#             not_selected_score.append(unselected_paths_for_class[0])
#             new_list[idx][jdx].append((selected_paths_for_class, jdx))
#
#     print(new_list)
#     print(score_)
#     print(not_selected_list)
#     print(not_selected_score)
#     return new_list


def calculate_similarity_3(label_metric_dict, training_metric_label_dict):
    # This list will store lists of file paths for each class
    selected_paths = [[] for _ in range(len(label_metric_dict))]

    # Loop over each class
    for true_label, label_metrics_list in label_metric_dict.items():
        # Skip this class if it doesn't exist in training_metric_label_dict

        if true_label not in training_metric_label_dict:
            continue

        # Loop over each sample in the unlabeled data for this class
        for training_metric, path in training_metric_label_dict[true_label]:
            # Normalize the metrics for unlabeled data
            training_metric = [metric / metric.norm(dim=-1, keepdim=True) for metric in training_metric]

            # min_final_score = float('inf')

            current_score = 0
            previous_indices = None
            # Loop over each set of metrics for this class in the labeled data
            for label_metrics in label_metrics_list:
                # Normalize the metrics for labeled data
                label_metrics = [metric / metric.norm(dim=-1, keepdim=True) for metric in label_metrics]

                # Calculate scores
                scores = [label_metric @ train_metric.transpose(-1, -2) for label_metric, train_metric in
                          zip(label_metrics, training_metric)]

                # Initialize the previous indices to None (this will be updated after the first iteration)


                for i, score in enumerate(scores):
                    # Calculate max along last dimension
                    node_max, node_idx = score.max(dim=-1)

                    # Get the indices that would sort the max values
                    sorted_indices = torch.argsort(node_max, descending=True)

                    # Get the top 70% indices
                    top_70_percent = int(0.5 * len(sorted_indices))
                    top_indices = sorted_indices[:top_70_percent]

                    if i == 0:
                        # If this is the first iteration, record the top indices
                        previous_indices = top_indices
                    else:
                        # For subsequent iterations, find the intersection of top indices with previous indices
                        previous_indices = torch.tensor(
                            list(set(previous_indices.tolist()).intersection(set(top_indices.tolist()))))

                # current_score.append([previous_indices, path])
                # if previous_indices < current_score:
                #     current_score = previous_indices
                current_score += len(previous_indices)
            selected_paths[true_label].append([current_score / len(label_metrics_list), path])

    new_list = []

    # loop through the list of classes
    for paths in selected_paths:
        # sort the list in descending order (highest first)
        sorted_paths = sorted(paths, key=lambda x: x[0], reverse=True)

        # select the 8 highest and 2 smallest and store their paths
        selected_paths_for_class = sorted_paths[:9]

        # extract the paths only
        selected_paths_for_class = [path[1] for path in selected_paths_for_class]

        new_list.append(selected_paths_for_class)

    # new_list now contains the paths associated with the 8 highest and 2 smallest values for each class.

    return new_list

def calculate_similarity(label_metric_dict, training_metric_label_dict):
    # This list will store lists of file paths for each class
    selected_paths = [[] for _ in range(len(label_metric_dict))]

    # Loop over each class
    for true_label, label_metrics_list in label_metric_dict.items():
        # Skip this class if it doesn't exist in training_metric_label_dict
        if true_label not in training_metric_label_dict:
            continue

        # Loop over each sample in the unlabeled data for this class
        for training_metric, path in training_metric_label_dict[true_label]:
            # Normalize the metrics for unlabeled data
            training_metric = [metric / metric.norm(dim=-1, keepdim=True) for metric in training_metric]

            min_final_score = float('inf')

            # Loop over each set of metrics for this class in the labeled data
            for label_metrics in label_metrics_list:
                # Normalize the metrics for labeled data
                label_metrics = [metric / metric.norm(dim=-1, keepdim=True) for metric in label_metrics]

                # Calculate scores
                scores = [label_metric @ train_metric.transpose(-1, -2) for label_metric, train_metric in
                          zip(label_metrics, training_metric)]

                # Get the min over one dimension
                scores = [score.max(dim=-1, keepdim=True)[0] for score in scores]

                # Sum over one dimension
                scores = [score.sum(dim=-1, keepdim=True) for score in scores]

                # Sum over all metrics to get final score
                final_score = sum(scores).sum().item()

                # If this final score is smaller than the current minimum, update the minimum
                if final_score < min_final_score:
                    min_final_score = final_score

            # Append the path and minimum final score to the list for this class
            selected_paths[true_label].append((min_final_score, path))

        # Sort the list for this class by score in ascending order
        selected_paths[true_label].sort(key=lambda x: x[0])

    # Only keep the paths, not the scores
    selected_paths = [[path for score, path in class_list] for class_list in selected_paths]

    return selected_paths


def calculate_similarity_v2(label_metric_dict, training_metric_label_dict, threshold=0.55, top_k=4):
    # This list will store lists of file paths for each class
    selected_paths = [[] for _ in range(len(label_metric_dict))]
    all_scores = [[] for _ in range(len(label_metric_dict))]
    # Loop over each class
    max_self_scores = [[] for _ in range(len(label_metric_dict))]
    for true_label, label_metrics_list in label_metric_dict.items():
        # Skip this class if it doesn't exist in training_metric_label_dict
        if true_label not in training_metric_label_dict:
            continue


        # calculate max_self_scores
        for label_metrics in label_metrics_list:
            label_metrics = [metric / metric.norm(dim=-1, keepdim=True) for metric in label_metrics]
            self_scores = [label_metric @ label_metric.transpose(-1, -2) for label_metric in label_metrics]
            max_self_scores[true_label].append(self_scores)


        # Loop over each sample in the unlabeled data for this class
        for training_metric, path in training_metric_label_dict[true_label]:
            # Normalize the metrics for unlabeled data
            training_metric = [metric / metric.norm(dim=-1, keepdim=True) for metric in training_metric]


            # Loop over each set of metrics for this class in the labeled data
            for label_metrics in label_metrics_list:
                # Normalize the metrics for labeled data
                label_metrics = [metric / metric.norm(dim=-1, keepdim=True) for metric in label_metrics]

                # Calculate scores
                scores = [label_metric @ train_metric.transpose(-1, -2) for label_metric, train_metric in
                          zip(label_metrics, training_metric)]

                # # calculate cosine similarity
                # cosine_similarity_scores = []
                # for i in range(len(max_self_scores)):
                #     cosine_similarity = torch.nn.functional.cosine_similarity(max_self_scores[i], scores[i], dim=0)
                #     cosine_similarity_scores.append(cosine_similarity)
                all_scores[true_label].append([scores, path])

    # starting calculate the cons similarity
    current_label_similar = []
    for true_label in range(len(max_self_scores)):
        per_label = []
        for needed_training in all_scores[true_label]:
            cosine_scores = []
            for element in max_self_scores[true_label]:
                current_simi = []
                for i in range(len(element)):
                    cosine_similarity = torch.nn.functional.cosine_similarity(needed_training[0][i], element[i], dim=0)
                    current_simi.append(cosine_similarity)
                cosine_scores.append([current_simi, needed_training[1]])
            per_label.append(cosine_scores)
        current_label_similar.append(per_label)


    return selected_paths


def uncertainty_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        entropy_dict = {i: {"entropy": [], "path": []} for i in
                        range(opt.NUM_CLASSES)}  # Initialize the dictionary to store entropy and path for each class

        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, k_metrics, features = engine.model(B, V, num_views, inputs)

            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)  # Calculate softmax for entropy computation
            entropy = -torch.sum(softmax_outputs * torch.log2(softmax_outputs + 1e-6),
                                 dim=1)  # Compute entropy for each sample

            prediction = torch.max(outputs, 1)[1]
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()  # Get the class index
                entropy_dict[class_index]["entropy"].append(entropy[i].item())  # Append the entropy
                entropy_dict[class_index]["path"].append(train_path[i])  # Append the corresponding train path

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(opt.NUM_CLASSES):
            max_entropy_index = torch.argmax(
                torch.tensor(entropy_dict[class_index]["entropy"]))  # Get index of maximum entropy
            highest_entropy_path = entropy_dict[class_index]["path"][max_entropy_index]  # Get corresponding path

            # Append the highest_entropy_path to the corresponding sublist in old_index_train
            old_index_train[class_index].append(highest_entropy_path)

            # Remove the highest_entropy_path from the sublist in old_index_not_train at class_index
            if highest_entropy_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(highest_entropy_path)
        # print(old_index_train)
        return old_index_train, old_index_not_train

def uncertainty_sampling_one_label_multi_Ob(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        entropy_dict = [{} for _ in range(opt.nb_classes)]

        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            # object_class 是 path + model id
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)

            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            entropy = -torch.sum(softmax_outputs * torch.log2(softmax_outputs + 1e-6), dim=1)




            for i in range(true_labels.size(0)):
                # continue
                true_label = true_labels[i].item()
                path = train_path[i]
                current_entropy = entropy[i]
                if object_class[i].item() not in entropy_dict[true_label]:
                    entropy_dict[true_label][object_class[i].item()] = []
                entropy_dict[true_label][object_class[i].item()].append([current_entropy, path])
                # class_index = transform_targets[i].item()
                # entropy_dict[class_index]["entropy"].append(entropy[i].item())
                # entropy_dict[class_index]["path"].append(train_path[i])
            del inputs, targets, outputs, features_k, features, true_labels
            torch.cuda.empty_cache()


        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(len(entropy_dict)):
            selected_path = entropy_dict[class_index]
            for object_class, value_ in selected_path.items():
                sorted_entropy = sorted(value_, key=lambda x: x[0], reverse=True)
                max_entropy = sorted_entropy[0][0]
                current_path = sorted_entropy[0][1]




                old_index_train[class_index][0][object_class].append((current_path, object_class))

                for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                    x1 = current_path
                    x2 = old_index_not_train[class_index][0][object_class][jdx][0]
                    if x1 == x2:
                        del old_index_not_train[class_index][0][object_class][jdx]
                        break


        return old_index_train, old_index_not_train



def dissimilarity_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset, train_data):
    engine.model.eval()

    with torch.no_grad():
        feature_dict_train = {i: {"features": []} for i in range(opt.nb_classes)}
        for index, (label, image, num_views, marks) in enumerate(train_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            transform_targets = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)
            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()
                feature_dict_train[class_index]["features"].append(features[i].detach().cpu().numpy())

    with torch.no_grad():
        feature_dict = {i: {"features": [], "path": []} for i in range(opt.nb_classes)}

        # First pass: collect features and paths
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)

            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()
                feature_dict[class_index]["features"].append(features[i].detach().cpu().numpy())
                feature_dict[class_index]["path"].append(train_path[i])

        dissimilarity_dict = {i: {"dissimilarity": [], "path": []} for i in range(opt.nb_classes)}

        # Second pass: compute dissimilarities
        for class_index in range(opt.nb_classes):
            if feature_dict[class_index]["features"]:
                # Normalize features before computing mean
                normalized_features_train = normalize(feature_dict_train[class_index]["features"])
                normalized_features = normalize(feature_dict[class_index]["features"])

                mean_features_train = np.mean(normalized_features_train, axis=0)  # Compute mean features of train data
                for i, feature in enumerate(normalized_features):
                    dissimilarity = cosine(feature, mean_features_train)  # Compute dissimilarity using cosine distance
                    dissimilarity_dict[class_index]["dissimilarity"].append(dissimilarity)
                    dissimilarity_dict[class_index]["path"].append(feature_dict[class_index]["path"][i])

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        # Third pass: select most dissimilar samples
        for class_index in range(opt.nb_classes):
            if dissimilarity_dict[class_index]["dissimilarity"]:
                max_dissimilarity_index = np.argmax(dissimilarity_dict[class_index]["dissimilarity"])
                most_dissimilar_path = dissimilarity_dict[class_index]["path"][max_dissimilarity_index]

                old_index_train[class_index].append(most_dissimilar_path)

                if most_dissimilar_path in old_index_not_train[class_index]:
                    old_index_not_train[class_index].remove(most_dissimilar_path)
        print(old_index_train)
        return old_index_train, old_index_not_train

def dissimilarity_sampling_object_wise(opt, engine, train_dataset, unlabeled_data, labeled_data, train_data):
    engine.model.eval()

    with torch.no_grad():
        feature_dict_train = [{} for _ in range(opt.nb_classes)]
        for index, (label, image, num_views, object_class) in enumerate(labeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)
            # features = F.max_pool2d(features, kernel_size=8, stride=8)
            # features = features.mean(dim=1).squeeze(0)
            # features = features.reshape(-1)
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()
                if object_class[i].item() not in feature_dict_train[true_label]:
                    feature_dict_train[true_label][object_class[i].item()] = {"features": []}
                feature_dict_train[true_label][object_class[i].item()]["features"].append(
                    features[i].detach().cpu().numpy())
            del inputs, targets, outputs, features_k, features, true_labels
            torch.cuda.empty_cache()

    with torch.no_grad():
        feature_dict = [{} for _ in range(opt.nb_classes)]
        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)
            # features = F.max_pool2d(features, kernel_size=8, stride=8)
            # features = features.mean(dim=1).squeeze(0)
            # features = features.reshape(-1)
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()
                if object_class[i].item() not in feature_dict[true_label]:
                    feature_dict[true_label][object_class[i].item()] = {"features": [], "path": []}
                feature_dict[true_label][object_class[i].item()]["features"].append(features[i].detach().cpu().numpy())
                feature_dict[true_label][object_class[i].item()]["path"].append(train_path[i])
            del inputs, targets, outputs, features_k, features, true_labels
            torch.cuda.empty_cache()

        dissimilarity_dict = [{} for _ in range(opt.nb_classes)]
        for class_index in range(opt.nb_classes):
            for object_class in feature_dict[class_index]:
                if feature_dict[class_index][object_class]["features"]:
                    normalized_features_train = normalize(np.array(feature_dict_train[class_index][object_class]["features"]))
                    normalized_features = normalize(np.array(feature_dict[class_index][object_class]["features"]))
                    mean_features_train = np.mean(normalized_features_train, axis=0)
                    for i, feature in enumerate(normalized_features):
                        dissimilarity = cosine(feature, mean_features_train)
                        if object_class not in dissimilarity_dict[class_index]:
                            dissimilarity_dict[class_index][object_class] = {"dissimilarity": [], "path": []}
                        dissimilarity_dict[class_index][object_class]["dissimilarity"].append(dissimilarity)
                        dissimilarity_dict[class_index][object_class]["path"].append(feature_dict[class_index][object_class]["path"][i])

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(opt.nb_classes):
            for object_class in dissimilarity_dict[class_index]:
                if dissimilarity_dict[class_index][object_class]["dissimilarity"]:
                    max_dissimilarity_index = np.argmax(dissimilarity_dict[class_index][object_class]["dissimilarity"])
                    most_dissimilar_path = dissimilarity_dict[class_index][object_class]["path"][max_dissimilarity_index]

                    old_index_train[class_index][0][object_class].append((most_dissimilar_path, object_class))

                    for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                        if most_dissimilar_path == old_index_not_train[class_index][0][object_class][jdx][0]:
                            del old_index_not_train[class_index][0][object_class][jdx]
                            break
        # print(old_index_train)
        return old_index_train, old_index_not_train

# def random_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
#     engine.model.eval()
#     with torch.no_grad():
#         entropy_dict = {i: {"entropy": [], "path": []} for i in
#                         range(44)}  # Initialize the dictionary to store entropy and path for each class
#
#         for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
#             inputs = Variable(image).to(engine.device)
#             targets = Variable(label).to(engine.device)
#             B, V, C, H, W = inputs.shape
#             inputs = inputs.view(-1, C, H, W)
#             outputs, features, utilization = engine.model(B, V, num_views, inputs)
#
#             softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)  # Calculate softmax for entropy computation
#             entropy = -torch.sum(softmax_outputs * torch.log2(softmax_outputs + 1e-6),
#                                  dim=1)  # Compute entropy for each sample
#
#             prediction = torch.max(outputs, 1)[1]
#             transform_targets = torch.max(targets, 1)[1]
#
#             for i in range(len(transform_targets)):
#                 class_index = transform_targets[i].item()  # Get the class index
#                 entropy_dict[class_index]["entropy"].append(entropy[i].item())  # Append the entropy
#                 entropy_dict[class_index]["path"].append(train_path[i])  # Append the corresponding train path
#
#         old_index_train = train_dataset.selected_ind_train
#         old_index_not_train = train_dataset.unselected_ind_train
#
#         for class_index in range(44):
#             max_entropy_index = torch.argmax(
#                 torch.tensor(entropy_dict[class_index]["entropy"]))  # Get index of maximum entropy
#             highest_entropy_path = entropy_dict[class_index]["path"][max_entropy_index]  # Get corresponding path
#
#             # Append the highest_entropy_path to the corresponding sublist in old_index_train
#             old_index_train[class_index].append(highest_entropy_path)
#
#             # Remove the highest_entropy_path from the sublist in old_index_not_train at class_index
#             if highest_entropy_path in old_index_not_train[class_index]:
#                 old_index_not_train[class_index].remove(highest_entropy_path)
#
#         return old_index_train, old_index_not_train


def random_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        path_dict = {i: [] for i in range(44)}  # Initialize the dictionary to store paths for each class

        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            targets = Variable(label).to(engine.device)
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()  # Get the class index
                path_dict[class_index].append(train_path[i])  # Append the corresponding train path

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(44):
            for idx in range(2):
                if path_dict[class_index]:  # Check if there are available paths for the class
                    selected_path = random.choice(path_dict[class_index])  # Get random path

                    # Append the selected_path to the corresponding sublist in old_index_train
                    old_index_train[class_index].append(selected_path)

                    # Remove the selected_path from the sublist in old_index_not_train at class_index
                    if selected_path in old_index_not_train[class_index]:
                        old_index_not_train[class_index].remove(selected_path)
        # print(old_index_train)
        # print(len(old_index_train[0]))
        return old_index_train, old_index_not_train

def random_sampling_object_wise(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        path_dict = [{} for _ in range(opt.nb_classes)]  # Initialize the dictionary to store paths for each class

        for index, (label, image, num_views, object_class, train_path) in enumerate(unlabeled_data):
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]

            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Get the class index
                if object_class[i].item() not in path_dict[true_label]:
                    path_dict[true_label][object_class[i].item()] = []
                path_dict[true_label][object_class[i].item()].append(train_path[i])  # Append the corresponding train path

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(len(path_dict)):
            for object_class in path_dict[class_index]:
                if path_dict[class_index][object_class]:  # Check if there are available paths for the class
                    selected_path = random.choice(path_dict[class_index][object_class])  # Get random path

                    # Append the selected_path to the corresponding sublist in old_index_train
                    old_index_train[class_index][0][object_class].append((selected_path, object_class))

                    # Remove the selected_path from the sublist in old_index_not_train at class_index
                    for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                        if selected_path == old_index_not_train[class_index][0][object_class][jdx][0]:
                            del old_index_not_train[class_index][0][object_class][jdx]
                            break


        return old_index_train, old_index_not_train



def LfOSA(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    path_dict = {i: [] for i in range(44)}
    activation_dict = {i: [] for i in range(44)}

    with torch.no_grad():
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)

            v_ij, predicted = outputs.max(1)
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()
                path_dict[class_index].append(train_path[i])
                activation_dict[class_index].append(np.array(v_ij.data.cpu())[i])

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(44):
        if path_dict[class_index]:  # Check if there are available paths for the class
            activation_values = np.array(activation_dict[class_index])
            if len(activation_values) < 2:
                continue

            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(activation_values.reshape(-1, 1))
            prob = gmm.predict_proba(activation_values.reshape(-1, 1))
            prob = prob[:, gmm.means_.argmax()]

            selected_index = np.argmax(prob)
            selected_path = path_dict[class_index][selected_index]

            old_index_train[class_index].append(selected_path)
            if selected_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(selected_path)

    return old_index_train, old_index_not_train


# def LfOSA(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
#     engine.model.eval()
#     path_dict = {i: [] for i in range(44)}
#     activation_dict = {i: [] for i in range(44)}
#
#     with torch.no_grad():
#         for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
#             inputs = Variable(image).to(engine.device)
#             targets = Variable(label).to(engine.device)
#             B, V, C, H, W = inputs.shape
#             inputs = inputs.view(-1, C, H, W)
#             outputs, features, utilization = engine.model(B, V, num_views, inputs)
#
#             v_ij, predicted = outputs.max(1)
#             transform_targets = torch.max(targets, 1)[1]
#
#             for i in range(len(transform_targets)):
#                 class_index = transform_targets[i].item()
#                 path_dict[class_index].append(train_path[i])
#                 activation_dict[class_index].append(np.array(v_ij.data.cpu())[i])
#
#     old_index_train = train_dataset.selected_ind_train
#     old_index_not_train = train_dataset.unselected_ind_train
#
#     for class_index in range(44):
#         if path_dict[class_index]:  # Check if there are available paths for the class
#             activation_values = np.array(activation_dict[class_index])
#             if len(activation_values) < 2:
#                 continue
#
#             gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
#             gmm.fit(activation_values.reshape(-1, 1))
#             prob = gmm.predict_proba(activation_values.reshape(-1, 1))
#             prob = prob[:, gmm.means_.argmax()]
#
#             selected_index = np.argmax(prob)
#             selected_path = path_dict[class_index][selected_index]
#
#             old_index_train[class_index].append(selected_path)
#             if selected_path in old_index_not_train[class_index]:
#                 old_index_not_train[class_index].remove(selected_path)
#
#     return old_index_train, old_index_not_train


def bayesian_generative_active_learning(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    path_dict = {i: {"uncertainty": [], "path": []} for i in
                 range(opt.nb_classes)}  # Initialize the dictionary to store paths and uncertainties for each class

    with torch.no_grad():
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)

            _, predicted = outputs.max(1)
            transform_targets = torch.max(targets, 1)[1]
            proba_out = torch.nn.functional.softmax(outputs, dim=1)
            proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

            uncertainty = 1 - proba_out.squeeze().cpu().numpy()
            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()  # Get the class index
                path_dict[class_index]["uncertainty"].append(uncertainty[i])  # Append the uncertainty
                path_dict[class_index]["path"].append(train_path[i])  # Append the corresponding train path

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(opt.nb_classes):
        if path_dict[class_index]["path"]:  # Check if there are available paths for the class
            # sort paths by their uncertainties and select the one with the highest uncertainty
            sorted_indices = np.argsort(-np.array(path_dict[class_index]["uncertainty"]))
            selected_path = path_dict[class_index]["path"][sorted_indices[0]]

            old_index_train[class_index].append(selected_path)  # Add the selected path to the training set
            if selected_path in old_index_not_train[class_index]:  # Remove the selected path from the unlabeled set
                old_index_not_train[class_index].remove(selected_path)

    return old_index_train, old_index_not_train


def bayesian_generative_active_learning_object_wise(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    uncertainty_dict = [{} for _ in range(opt.nb_classes)]

    with torch.no_grad():
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)

            _, predicted = outputs.max(1)
            true_labels = torch.max(targets, 1)[1]
            proba_out = torch.nn.functional.softmax(outputs, dim=1)
            proba_out = torch.gather(proba_out, 1, predicted.unsqueeze(1))

            uncertainty = 1 - proba_out.squeeze().cpu().numpy()
            for i in range(len(true_labels)):
                true_label = true_labels[i].item()
                path = train_path[i]
                current_uncertainty = uncertainty[i]
                object_class = marks[
                    i].item()  # Assuming `marks` can be used similarly to `object_class` in the first function

                if object_class not in uncertainty_dict[true_label]:
                    uncertainty_dict[true_label][object_class] = []
                uncertainty_dict[true_label][object_class].append([current_uncertainty, path])
            del inputs, targets, outputs, features_k, features, true_labels
            torch.cuda.empty_cache()

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(len(uncertainty_dict)):
        selected_path = uncertainty_dict[class_index]
        for object_class, value_ in selected_path.items():
            sorted_uncertainty = sorted(value_, key=lambda x: x[0], reverse=True)
            max_uncertainty = sorted_uncertainty[0][0]
            current_path = sorted_uncertainty[0][1]

            old_index_train[class_index][0][object_class].append((current_path, object_class))

            for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                x1 = current_path
                x2 = old_index_not_train[class_index][0][object_class][jdx][0]
                if x1 == x2:
                    del old_index_not_train[class_index][0][object_class][jdx]
                    break

    return old_index_train, old_index_not_train


def compute_openmax_scores(activations, mavs, weibull_models, labels, known_class):
    openmax_scores = []
    for activation, label in zip(activations, labels):
        # For all classes, compute openmax score
        clipped_activation = np.clip(activation - mavs[label.item()], a_min=0, a_max=None)
        params = weibull_models[label.item()]["params"]
        w_score = 1 - np.exp(-(clipped_activation ** params[1]))
        score = w_score / (w_score + (1 - w_score + 1e-7) / known_class)
        openmax_scores.append(np.max(score))
    return openmax_scores


# Weibull CDF calculation
def weibull_cdf(x, params):
    return 1 - np.exp(-((x / params[0]) ** params[1]))

def open_max(opt, engine, labeled_dataset, train_dataset, unlabeled_data):
    engine.model.eval()

    with torch.no_grad():
        # First part: calculating the mavs and weibull_models using the labeled dataset
        features_dict = {i: [] for i in range(44)}

        for index, (label, image, num_views, marks) in enumerate(labeled_dataset):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)
            predicted = torch.max(outputs, 1)[1]
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()
                features_dict[class_index].append(features[i].detach().cpu().numpy())

        mavs = {c: np.mean(features, axis=0) for c, features in features_dict.items()}

        weibull_models = {c: {"distances": [], "params": [], "mean_distance": 0, "inv_std_distance": 0} for c in
                          range(44)}

        def weibull_pdf(x, shape, scale):
            return (shape / scale) * (x / scale) ** (shape - 1) * np.exp(- (x / scale) ** shape)

        def neg_log_likelihood(params):
            return -np.sum(np.log(weibull_pdf(distances_normalized, *params)))

        for c in range(44):
            distances = [distance.euclidean(f, mavs[c]) for f in features_dict[c]]
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            distances_normalized = (distances - mean_distance) / std_distance

            initial_guess = [1, 1]
            bounds = [(0.1, None), (0.1, None)]
            result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)

            weibull_models[c]["distances"] = distances
            weibull_models[c]["mean_distance"] = mean_distance
            weibull_models[c]["inv_std_distance"] = 1 / std_distance
            weibull_models[c]["params"] = result.x

        # Second part: calculating the score using the unlabeled dataset
        score_index_pairs = []
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)
            transform_targets = torch.max(targets, 1)[1]

            openmax_scores = compute_openmax_scores(features.detach().cpu().numpy(), mavs, weibull_models,
                                                    transform_targets, 44)
            score_index_pairs.extend(list(zip(openmax_scores, train_path, transform_targets.cpu().numpy())))

        # Sort the pairs in descending order of scores
        sorted_pairs = []
        for class_index in range(44):
            class_pairs = [pair for pair in score_index_pairs if pair[2] == class_index]
            class_pairs.sort(key=lambda x: -x[0])
            if class_pairs:
                sorted_pairs.append(class_pairs[0])  # append only the pair with the highest score

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(44):
            if class_index < len(sorted_pairs):
                _, selected_path, _ = sorted_pairs[class_index]
                old_index_train[class_index].append(selected_path)
                if selected_path in old_index_not_train[class_index]:
                    old_index_not_train[class_index].remove(selected_path)

    return old_index_train, old_index_not_train

def open_max_object_wise(opt, engine, labeled_dataset, train_dataset, unlabeled_data):
    engine.model.eval()
    score_dict = [{} for _ in range(opt.nb_classes)]

    with torch.no_grad():
        # First part: calculating the mavs and weibull_models using the labeled dataset
        features_dict = {i: [] for i in range(opt.nb_classes)}

        for index, (label, image, num_views, marks) in enumerate(labeled_dataset):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features_k, features = engine.model(B, V, num_views, inputs)
            predicted = torch.max(outputs, 1)[1]
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()
                features_dict[class_index].append(features[i].detach().cpu().numpy())

        mavs = {c: np.mean(features, axis=0) for c, features in features_dict.items()}

        weibull_models = {c: {"distances": [], "params": [], "mean_distance": 0, "inv_std_distance": 0} for c in
                          range(opt.nb_classes)}

        def weibull_pdf(x, shape, scale):
            return (shape / scale) * (x / scale) ** (shape - 1) * np.exp(- (x / scale) ** shape)

        def neg_log_likelihood(params):
            return -np.sum(np.log(weibull_pdf(distances_normalized, *params)))

        for c in range(opt.nb_classes):
            distances = [distance.euclidean(f, mavs[c]) for f in features_dict[c]]
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            distances_normalized = (distances - mean_distance) / std_distance

            initial_guess = [1, 1]
            bounds = [(0.1, None), (0.1, None)]
            result = minimize(neg_log_likelihood, initial_guess, bounds=bounds)

            weibull_models[c]["distances"] = distances
            weibull_models[c]["mean_distance"] = mean_distance
            weibull_models[c]["inv_std_distance"] = 1 / std_distance
            weibull_models[c]["params"] = result.x

        # Second part: calculating the score using the unlabeled dataset
        with torch.no_grad():
            for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
                inputs = Variable(image).to(engine.device)
                targets = Variable(label).to(engine.device)
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features_k, features = engine.model(B, V, num_views, inputs)
                true_labels = torch.max(targets, 1)[1]

                openmax_scores = compute_openmax_scores(features.detach().cpu().numpy(), mavs, weibull_models,
                                                        true_labels, 44)

                for i in range(len(true_labels)):
                    true_label = true_labels[i].item()
                    path = train_path[i]
                    current_score = openmax_scores[i]
                    object_class = marks[
                        i].item()  # Assuming `marks` can be used similarly to `object_class` in the first function

                    if object_class not in score_dict[true_label]:
                        score_dict[true_label][object_class] = []
                    score_dict[true_label][object_class].append([current_score, path])

        # # Sort the pairs in descending order of scores
        # sorted_pairs = []
        # for class_index in range(opt.nb_classes):
        #     class_pairs = [pair for pair in score_index_pairs if pair[2] == class_index]
        #     class_pairs.sort(key=lambda x: -x[0])
        #     if class_pairs:
        #         sorted_pairs.append(class_pairs[0])  # append only the pair with the highest score

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(len(score_dict)):
            selected_path = score_dict[class_index]
            for object_class, value_ in selected_path.items():
                sorted_scores = sorted(value_, key=lambda x: x[0], reverse=True)
                max_score = sorted_scores[0][0]
                current_path = sorted_scores[0][1]

                old_index_train[class_index][0][object_class].append((current_path, object_class))

                for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                    x1 = current_path
                    x2 = old_index_not_train[class_index][0][object_class][jdx][0]
                    if x1 == x2:
                        del old_index_not_train[class_index][0][object_class][jdx]
                        break

        return old_index_train, old_index_not_train


def core_set(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    min_distances = [None] * opt.nb_classes
    already_selected = [[] for _ in range(opt.nb_classes)]
    features = [[] for _ in range(opt.nb_classes)]
    indices = [[] for _ in range(opt.nb_classes)]
    labels = [[] for _ in range(opt.nb_classes)]

    for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
        inputs = Variable(image).to(engine.device)
        targets = Variable(label).to(engine.device)
        B, V, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        outputs, features_k, feature = engine.model(B, V, num_views, inputs)

        transform_targets = torch.max(targets, 1)[1]
        for i in range(len(transform_targets)):
            class_index = transform_targets[i].item()
            features[class_index].append((feature[i].detach().cpu().numpy(), train_path[i]))
            indices[class_index].append(train_path[i])
            labels[class_index].append(class_index)

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(opt.nb_classes):  # instead of opt.query_batch
        feature_array = np.array([f[0] for f in features[class_index]])
        if not already_selected[class_index]:
            ind = np.random.choice(len(features[class_index]))
        else:
            ind = np.argmax(min_distances[class_index])

        dist = euclidean_distances(feature_array, feature_array[ind].reshape(1, -1))

        if min_distances[class_index] is None:
            min_distances[class_index] = dist
        else:
            min_distances[class_index] = np.minimum(min_distances[class_index], dist)

        already_selected[class_index].append(ind)

        selected_path = indices[class_index][ind]
        label = labels[class_index][ind]

        old_index_train[label].append(selected_path)
        if selected_path in old_index_not_train[label]:
            old_index_not_train[label].remove(selected_path)

    return old_index_train, old_index_not_train


def core_set_object_wise(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    features = [{} for _ in range(opt.nb_classes)]
    indices = [{} for _ in range(opt.nb_classes)]
    already_selected = [{} for _ in range(opt.nb_classes)]
    min_distances = [{} for _ in range(opt.nb_classes)]

    for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
        inputs = Variable(image).to(engine.device)
        targets = Variable(label).to(engine.device)
        B, V, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        outputs, features_k, feature = engine.model(B, V, num_views, inputs)

        transform_targets = torch.max(targets, 1)[1]
        for i in range(len(transform_targets)):
            class_index = transform_targets[i].item()
            object_class = marks[i].item()
            if object_class not in features[class_index]:
                features[class_index][object_class] = []
                indices[class_index][object_class] = []
                already_selected[class_index][object_class] = []
                min_distances[class_index][object_class] = None
            features[class_index][object_class].append(feature[i].detach().cpu().numpy())
            indices[class_index][object_class].append(train_path[i])

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(opt.nb_classes):
        for object_class in features[class_index].keys():
            feature_array = np.array(features[class_index][object_class])
            if not feature_array.size:  # Skip if no features
                continue

            # Find the index that maximizes the minimum distance to all previously selected features
            if not already_selected[class_index][object_class]:
                ind = np.random.choice(len(feature_array))
            else:
                distances = euclidean_distances(feature_array, feature_array)
                min_distances[class_index][object_class] = np.min(
                    distances[:, already_selected[class_index][object_class]], axis=1)
                ind = np.argmax(min_distances[class_index][object_class])

            # Store the selected index for future distance calculations
            already_selected[class_index][object_class].append(ind)

            selected_path = indices[class_index][object_class][ind]

            old_index_train[class_index][0][object_class].append((selected_path, object_class))

            for jdx in range(len(old_index_not_train[class_index][0][object_class])):
                x1 = selected_path
                x2 = old_index_not_train[class_index][0][object_class][jdx][0]
                if x1 == x2:
                    del old_index_not_train[class_index][0][object_class][jdx]
                    break

    return old_index_train, old_index_not_train


def init_centers(X, labels, K):
    embs = torch.Tensor(X)
    embs = embs.cuda()

    # Group embeddings by class
    grouped_embs = {c: [] for c in range(K)}
    for i, (emb, label) in enumerate(zip(embs, labels)):
        grouped_embs[label].append(
            (emb, i))  # Store the embedding and index as a tuple in the 'grouped_embs' dictionary

    indsAll = []
    for c in range(K):
        class_embs = grouped_embs[c]
        if not class_embs:  # If no examples for this class, skip
            continue

        class_embs, indices = zip(*class_embs)  # Unzip the list of tuples into separate lists

        class_embs = torch.stack(class_embs)
        ind = torch.argmax(torch.norm(class_embs, 2, 1)).item()

        mu = [class_embs[ind]]
        centInds = [0.] * len(class_embs)
        cent = 0

        while len(mu) < len(class_embs):
            if len(mu) == 1:
                D2 = torch.cdist(mu[-1].view(1, -1), class_embs, 2)[0].cpu().numpy()
            else:
                newD = torch.cdist(mu[-1].view(1, -1), class_embs, 2)[0].cpu().numpy()
                for i in range(len(class_embs)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0:
                break

            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll:
                ind = customDist.rvs(size=1)[0]
            mu.append(class_embs[ind])
            indsAll.append(indices[ind])  # Append the index from 'indices' list
            cent += 1

        # Select one index from this class
        indsAll.append(indices[ind])  # Append the index from 'indices' list

    # Match the size of indsAll to labels
    indsAll = indsAll[:len(labels)]

    return indsAll


def badge_sampling(opt, engine, train_dataset, labeled_data, unlabeled_data, train_data):
    engine.model.eval()
    embDim = 512
    nLab = 44
    len_unlabeled_data = len(unlabeled_data.dataset)
    len_labeled_data = len(train_data.dataset)
    embedding = np.zeros([len_unlabeled_data + len_labeled_data, embDim * nLab])

    S_ij = {}

    # Flatten the list of lists into a single list
    unselected_ind_train_flat = [item for sublist in train_dataset.unselected_ind_train for item in sublist]

    # Create new_index dictionary
    new_index = {i: path for i, path in enumerate(unselected_ind_train_flat)}
    counter = 0
    labels = []

    for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
        inputs = Variable(image).to(engine.device)
        targets = Variable(label).to(engine.device)
        B, V, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        outputs, out, utilization = engine.model(B, V, num_views, inputs)
        out = out.data.cpu().numpy()
        batchProbs = F.softmax(outputs, dim=1).data.cpu().numpy()
        maxInds = np.argmax(batchProbs, 1)
        targets = torch.max(targets, 1)[1]
        labels.extend(targets.cpu().numpy())
        for j in range(len(targets)):
            for c in range(nLab):
                if c == maxInds[j]:
                    embedding[counter][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                else:
                    embedding[counter][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            counter += 1

            v_ij, predicted = outputs.max(1)
            for i in range(len(predicted.data)):
                tmp_class = np.array(predicted.data.cpu())[i]
                tmp_index = counter
                tmp_label = np.array(targets.data.cpu())[i]
                tmp_value = np.array(v_ij.data.cpu())[i]

                if tmp_index not in S_ij:
                    S_ij[tmp_index] = []
                S_ij[tmp_index].append([tmp_class, tmp_value, tmp_label])
    labels = np.array(labels)
    embedding = torch.Tensor(embedding)
    queryIndex = init_centers(embedding, labels, 44)

    queryLabelArr = []
    for i in range(len(queryIndex)):
        queryLabelArr.append(S_ij[queryIndex[i]][0][2])

    queryLabelArr = np.array(queryLabelArr)

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(44):
        if queryIndex[class_index]:
            selected_path = queryIndex[class_index]
            old_index_train[class_index].append(selected_path)
            if selected_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(selected_path)

    return old_index_train, old_index_not_train


def certainty_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        certainty_dict = {i: {"certainty": [], "path": []} for i in
                          range(44)}  # Initialize dictionary for certainty and path

        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)

            certainties = torch.softmax(outputs, dim=1).max(1).values.cpu().data  # Compute certainties

            prediction = torch.max(outputs, 1)[1]
            transform_targets = torch.max(targets, 1)[1]

            for i in range(len(transform_targets)):
                class_index = transform_targets[i].item()  # Get the class index
                certainty_dict[class_index]["certainty"].append(certainties[i].item())  # Append the certainty
                certainty_dict[class_index]["path"].append(train_path[i])  # Append the corresponding train path

        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

        for class_index in range(44):
            max_certainty_index = torch.argmax(
                torch.tensor(certainty_dict[class_index]["certainty"]))  # Get index of maximum certainty
            highest_certainty_path = certainty_dict[class_index]["path"][max_certainty_index]  # Get corresponding path

            # Append the highest_certainty_path to the corresponding sublist in old_index_train
            old_index_train[class_index].append(highest_certainty_path)

            # Remove the highest_certainty_path from the sublist in old_index_not_train at class_index
            if highest_certainty_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(highest_certainty_path)

        return old_index_train, old_index_not_train



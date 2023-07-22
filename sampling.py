import numpy as np
from torch.autograd import Variable
import torch
import random
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
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


def patch_based_selection(opt, engine, train_dataset, unlabeled_data, labeled_dataset, train_data,
                          unlabeled_sampling_labeled_data,
                          unlabeled_sampling_unlabeled_data):
    model_name = "vit_base_patch16_224"

    # Load a pretrained model
    model = timm.create_model(model_name, pretrained=True)
    #
    tome.patch.timm(model)
    model = model.to('cuda')
    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * opt.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(opt.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])
    # # Run the model with no reduction (should be the same as before)
    model.r = 0


    unlabeled_dataset = Unlabeled_Dataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'patched unlabeled',
                                          opt.MAX_NUM_VIEWS, unlabeled_sampling_labeled_data,
                                          unlabeled_sampling_unlabeled_data, transform)
    unlabeled_data = DataLoader(unlabeled_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS,
                                shuffle=True,
                                pin_memory=True, worker_init_fn=tool.seed_worker)

    labeled_dataset = Unlabeled_Dataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'patched labeled',
                                        opt.MAX_NUM_VIEWS, unlabeled_sampling_labeled_data,
                                        unlabeled_sampling_unlabeled_data, transform)
    labeled_data = DataLoader(labeled_dataset, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS,
                              shuffle=False,
                              pin_memory=True, worker_init_fn=tool.seed_worker)
    # batch = next(iter(labeled_data))
    engine.model.eval()
    with torch.no_grad():
    # opt.NUM_CLASSES
        label_metric_dict = {}

        for index, (label, image, num_views, marks) in enumerate(labeled_data):
            image = image.squeeze(1)
            inputs = Variable(image).to(engine.device)
            # model(inputs)
            model(inputs)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]
            metrics = model._tome_info["metric"]
            # print(len(metrics))
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = [m_list[i] for m_list in metrics]
                # Check if the label exists in the dictionary
                if true_label not in label_metric_dict:
                    label_metric_dict[true_label] = []
                # Add new_metric_list to dictionary
                label_metric_dict[true_label].append(new_metric_list)

        training_metric_label_dict = {}

        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            image = image.squeeze(1)
            inputs = Variable(image).to(engine.device)
            model(inputs)
            targets = Variable(label).to(engine.device)
            true_labels = torch.max(targets, 1)[1]  # This is now a tensor of labels for the batch

            metrics = model._tome_info["metric"]  # This should be a tensor of metrics for the batch

            # Loop over the batch
            for i in range(true_labels.size(0)):
                true_label = true_labels[i].item()  # Convert tensor to Python scalar
                # For each list in metrics, take the i-th element and add to a new list
                new_metric_list = [m_list[i] for m_list in metrics]
                path = train_path[i]  # Get the train path for this image
                if true_label not in training_metric_label_dict:
                    training_metric_label_dict[true_label] = []
                # Add new_metric_list and train_path to dictionary
                training_metric_label_dict[true_label].append([new_metric_list, path])

        selected_path = calculate_similarity(label_metric_dict, training_metric_label_dict)



        old_index_train = train_dataset.selected_ind_train
        old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(len(selected_path)):
        # Get the path of the least similar image for this class
        least_similar_path = selected_path[class_index][
            0]  # Assuming the paths are sorted in ascending order of similarity

        # Append the least_similar_path to the corresponding sublist in old_index_train
        old_index_train[class_index].append(least_similar_path)

        # Remove the least_similar_path from the sublist in old_index_not_train at class_index
        # Assuming old_index_not_train is a list of lists where each sublist corresponds to a class and contains the paths of the images for that class
        if least_similar_path in old_index_not_train[class_index]:
            old_index_not_train[class_index].remove(least_similar_path)

    return old_index_train, old_index_not_train


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
                scores = [label_metric @ train_metric.transpose(-1, -2) for label_metric, train_metric in zip(label_metrics, training_metric)]

                # Get the min over one dimension
                scores = [score.min(dim=-1, keepdim=True)[0] for score in scores]

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














def uncertainty_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    with torch.no_grad():
        entropy_dict = {i: {"entropy": [], "path": []} for i in
                        range(44)}  # Initialize the dictionary to store entropy and path for each class

        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)

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

        for class_index in range(44):
            max_entropy_index = torch.argmax(
                torch.tensor(entropy_dict[class_index]["entropy"]))  # Get index of maximum entropy
            highest_entropy_path = entropy_dict[class_index]["path"][max_entropy_index]  # Get corresponding path

            # Append the highest_entropy_path to the corresponding sublist in old_index_train
            old_index_train[class_index].append(highest_entropy_path)

            # Remove the highest_entropy_path from the sublist in old_index_not_train at class_index
            if highest_entropy_path in old_index_not_train[class_index]:
                old_index_not_train[class_index].remove(highest_entropy_path)

        return old_index_train, old_index_not_train


def dissimilarity_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset, train_data):
    engine.model.eval()

    with torch.no_grad():
        feature_dict_train = {i: {"features": []} for i in range(44)}
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
        feature_dict = {i: {"features": [], "path": []} for i in range(44)}

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

        dissimilarity_dict = {i: {"dissimilarity": [], "path": []} for i in range(44)}

        # Second pass: compute dissimilarities
        for class_index in range(44):
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
        for class_index in range(44):
            if dissimilarity_dict[class_index]["dissimilarity"]:
                max_dissimilarity_index = np.argmax(dissimilarity_dict[class_index]["dissimilarity"])
                most_dissimilar_path = dissimilarity_dict[class_index]["path"][max_dissimilarity_index]

                old_index_train[class_index].append(most_dissimilar_path)

                if most_dissimilar_path in old_index_not_train[class_index]:
                    old_index_not_train[class_index].remove(most_dissimilar_path)

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
            if path_dict[class_index]:  # Check if there are available paths for the class
                selected_path = random.choice(path_dict[class_index])  # Get random path

                # Append the selected_path to the corresponding sublist in old_index_train
                old_index_train[class_index].append(selected_path)

                # Remove the selected_path from the sublist in old_index_not_train at class_index
                if selected_path in old_index_not_train[class_index]:
                    old_index_not_train[class_index].remove(selected_path)

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
                 range(44)}  # Initialize the dictionary to store paths and uncertainties for each class

    with torch.no_grad():
        for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
            inputs = Variable(image).to(engine.device)
            targets = Variable(label).to(engine.device)
            B, V, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            outputs, features, utilization = engine.model(B, V, num_views, inputs)

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

    for class_index in range(44):
        if path_dict[class_index]["path"]:  # Check if there are available paths for the class
            # sort paths by their uncertainties and select the one with the highest uncertainty
            sorted_indices = np.argsort(-np.array(path_dict[class_index]["uncertainty"]))
            selected_path = path_dict[class_index]["path"][sorted_indices[0]]

            old_index_train[class_index].append(selected_path)  # Add the selected path to the training set
            if selected_path in old_index_not_train[class_index]:  # Remove the selected path from the unlabeled set
                old_index_not_train[class_index].remove(selected_path)

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


def core_set(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    min_distances = [None] * 44
    already_selected = [[] for _ in range(44)]
    features = [[] for _ in range(44)]
    indices = [[] for _ in range(44)]
    labels = [[] for _ in range(44)]

    for index, (label, image, num_views, marks, train_path) in enumerate(unlabeled_data):
        inputs = Variable(image).to(engine.device)
        targets = Variable(label).to(engine.device)
        B, V, C, H, W = inputs.shape
        inputs = inputs.view(-1, C, H, W)
        outputs, feature, utilization = engine.model(B, V, num_views, inputs)

        transform_targets = torch.max(targets, 1)[1]
        for i in range(len(transform_targets)):
            class_index = transform_targets[i].item()
            features[class_index].append((feature[i].detach().cpu().numpy(), train_path[i]))
            indices[class_index].append(train_path[i])
            labels[class_index].append(class_index)

    old_index_train = train_dataset.selected_ind_train
    old_index_not_train = train_dataset.unselected_ind_train

    for class_index in range(44):  # instead of opt.query_batch
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

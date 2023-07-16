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


def dissimilarity_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()

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
                mean_features = np.mean(feature_dict[class_index]["features"], axis=0)  # Compute mean features
                for i, feature in enumerate(feature_dict[class_index]["features"]):
                    dissimilarity = np.linalg.norm(feature - mean_features)  # Compute dissimilarity
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


def random_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
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



def bayesian_generative_active_learning(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    path_dict = {i: {"uncertainty": [], "path": []} for i in range(44)}  # Initialize the dictionary to store paths and uncertainties for each class

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
    return 1 - np.exp(-((x/params[0])**params[1]))


def open_max(opt, engine, labeled_dataset,train_dataset, unlabeled_data):

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

        weibull_models = {c: {"distances": [], "params": [], "mean_distance": 0, "inv_std_distance": 0} for c in range(44)}

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

            openmax_scores = compute_openmax_scores(features.detach().cpu().numpy(), mavs, weibull_models, transform_targets, 44)
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


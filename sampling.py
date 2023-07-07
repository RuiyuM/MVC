import numpy as np
from torch.autograd import Variable
import torch


def uncertainty_sampling(opt, engine, train_dataset, unlabeled_data, labeled_dataset):
    engine.model.eval()
    labeled_output = [[] for _ in range(opt.NUM_CLASSES)]
    unlabeled_output = [[] for _ in range(opt.NUM_CLASSES)]
    # with torch.no_grad():
    #     for index, (label, image, num_views, marks) in enumerate(labeled_data):
    #         inputs = Variable(image).to(engine.device)
    #         targets = Variable(label).to(engine.device)
    #         transform_targets = torch.max(targets, 1)[1]
    #         B, V, C, H, W = inputs.shape
    #         inputs = inputs.view(-1, C, H, W)
    #         outputs, features, utilization = engine.model(B, V, num_views, inputs)
    #         labeled_output[transform_targets].append(outputs)

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
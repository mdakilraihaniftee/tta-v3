import torch
import logging
import numpy as np
from typing import Union
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
import wandb



logger = logging.getLogger(__name__)

import numpy as np
import torch



def compute_tta_composite_score(accuracy_list, k=3, alpha=0.8, lambda_std=1.0):
    """
    Compute a composite score for Test-Time Adaptation performance.
    """
    # If input is a list of tensors, convert to CPU floats
    if isinstance(accuracy_list[0], torch.Tensor):
        acc = np.array([a.item() if a.device.type == 'cpu' else a.cpu().item() for a in accuracy_list])
    else:
        acc = np.array(accuracy_list)

    # Moving average
    ma = np.convolve(acc, np.ones(k)/k, mode='valid')
    
    # Threshold for plateau
    threshold = alpha * np.max(acc)
    
    # Time to plateau (first index where MA >= threshold)
    plateau_indices = np.where(ma >= threshold)[0]
    time_to_plateau = int(plateau_indices[0]) if len(plateau_indices) > 0 else len(ma)

    # Average Positive Slope (APS)
    diffs = np.diff(ma)
    positive_diffs = diffs[diffs > 0]
    avg_positive_slope = float(np.mean(positive_diffs)) if len(positive_diffs) > 0 else 0.0

    # Stability (STD of MA)
    stability_std = float(np.std(ma))

    # Composite Score
    composite_score = (avg_positive_slope / (time_to_plateau + 1e-5)) - lambda_std * stability_std

    return composite_score


def calculate_classwise_accuracy(labels, predictions, num_classes):
    # Initialize a list to store accuracy for each class
    class_accuracies = []
    
    # Loop through each class
    for class_label in range(num_classes):
        # Get all indices where the label is equal to the current class
        label_indices = (labels == class_label)
        model_indices = (predictions == class_label)
        
        # Check if class is missing in either label or model tensor
        if label_indices.sum() == 0 or model_indices.sum() == 0:
            class_accuracies.append(None)
        else:
            # Calculate accuracy for this class: number of correct predictions / total occurrences
            correct_predictions = (labels[label_indices] == predictions[label_indices]).sum().item()
            total_predictions = label_indices.sum().item()
            accuracy = correct_predictions / total_predictions
            class_accuracies.append(accuracy)
    
    return class_accuracies


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):

    num_correct = 0.
    num_samples = 0
    accu_li = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            

            #new
            attack_sorted = False
            if attack_sorted:
                # Sort the labels and get the sorting indices
                sorted_indices = torch.argsort(labels)

                # Use the sorted indices to reorder imgs and labels
                imgs = imgs[sorted_indices]
                labels = labels[sorted_indices]
                #print(labels)


            

            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device), labels.to(device))
            predictions = output.argmax(1)
            
            
            # if i%10==0:
            #     print("labels")
            #     print(labels)
            #     print("predictions")
            #     print(predictions)

            # Calculate class-wise accuracy
            accuracies = calculate_classwise_accuracy(labels.to(device), predictions, num_classes=10)
            # Log class-wise accuracy to WandB
            class_accuracies = {f'class_{i}_accuracy': accuracy if accuracy is not None else None for i, accuracy in enumerate(accuracies)}
            wandb.log(class_accuracies)


            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            num_c = (predictions == labels.to(device)).float().sum()
            acc = num_c/predictions.shape[0]
            accu_li.append(acc)
            # print(acc)
            wandb.log({"accuracy": acc})
            num_correct += num_c

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break
    
    c_score = compute_tta_composite_score(accu_li, k=3)

    accuracy = num_correct.item() / num_samples
    return accuracy, domain_dict, num_samples, c_score


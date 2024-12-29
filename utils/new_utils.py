import torch
<<<<<<< HEAD
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

=======
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange




import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.losses import (
    Entropy
)
ent = Entropy()
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
def plot_tsne(img_features, text_features, filename="tsne_plot.pdf"):
    # Detach tensors and convert to numpy arrays
    img_features_np = img_features.detach().cpu().numpy()
    text_features_np = text_features.detach().cpu().numpy()
    
    # Combine image and text features
    combined_features = np.concatenate((img_features_np, text_features_np), axis=0)
    
<<<<<<< HEAD
=======

    #new edit
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(combined_features)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:img_features_np.shape[0], 0], tsne_results[:img_features_np.shape[0], 1], label='Image Features')
    plt.scatter(tsne_results[img_features_np.shape[0]:, 0], tsne_results[img_features_np.shape[0]:, 1], label='Text Features')
    plt.legend()
    plt.title('t-SNE of Image and Text Features')
    
    # Save the plot as a PDF
    plt.savefig(filename)
    plt.show()

<<<<<<< HEAD
def plot_random_images( x, save_path="random_images.png"):
=======
def plot_random_images( x, save_path="random_images.png", indices = None):
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
        """
        Plots 12 random images from the given list `x` where each image is in the shape (1, 3, 224, 224).
        Saves the plotted images to the specified path.
        """
        # Select 12 random indices from the list `x`
<<<<<<< HEAD
        indices = np.random.choice(len(x), size=12, replace=False)
=======
        if indices is None:
            indices = np.random.choice(len(x), size=12, replace=False)
>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332
        selected_images = [x[i] for i in indices]

        # Set up the matplotlib figure
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle("Random Images", fontsize=16)

        for ax, img_tensor in zip(axes.flatten(), selected_images):
            # Remove batch dimension, move to CPU, and convert to numpy array
            img = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Display the image
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
<<<<<<< HEAD
        plt.show()
=======
        plt.show()

        return indices



def confident_wrong_pred(outputs, y):
    # Compute softmax probabilities
    logits = torch.nn.functional.softmax(outputs, dim=1)
    
    # Identify correct predictions
    correct = torch.argmax(logits, dim=1) == y
    
    # Calculate entropies for the batch
    outputs_ent = ent(outputs)
    
    # Filter entropies for incorrect predictions
    incorrect_entropies = outputs_ent[~correct]
    
    # Compute the average entropy for incorrect predictions
    if incorrect_entropies.numel() > 0:
        avg_entropy = incorrect_entropies.mean().item()
    else:
        avg_entropy = float('nan')  # Handle case with no incorrect predictions
    
    #print(f"Average entropy of incorrect predictions: {avg_entropy}")
    return avg_entropy


def confident_correct_pred(outputs, y):
    # Compute softmax probabilities
    logits = torch.nn.functional.softmax(outputs, dim=1)
    
    # Identify correct predictions
    correct = torch.argmax(logits, dim=1) == y
    
    # Calculate entropies for the batch
    outputs_ent = ent(outputs)
    
    # Filter entropies for incorrect predictions
    correct_entropies = outputs_ent[~correct]
    
    # Compute the average entropy for incorrect predictions
    if correct_entropies.numel() > 0:
        avg_entropy = correct_entropies.mean().item()
    else:
        avg_entropy = float('nan')  # Handle case with no incorrect predictions
    
    #print(f"Average entropy of correct predictions: {avg_entropy}")
    return avg_entropy

>>>>>>> d174a543d4d4f8bae0c0c46035b00c518c79a332

import torch
import json
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torchvision.transforms import v2
import random
import matplotlib.pyplot as plt
import shutil

from Models import define_network # esta al init de models
from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *

CUDA_LAUNCH_BLOCKING = 1
# TORCH_USE_CUDA_DSA = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='VIT_pretrained_emb_lowerlr_5cap_evenSimpler_harder_frozen')
    args = parser.parse_args()
    config = LoadConfig_baseline(args.test_name)
    num_imgs = 20


    root_dir = config['root_dir']
    folders = os.listdir(root_dir[:-1])
    folders_imgs = [folder for folder in folders if folder.endswith('epoch')]
    for folder_weights in folders_imgs:
        print(f"Analysing {folder_weights}")
        imgs_names_bleu2 = os.listdir(f"{root_dir}{folder_weights}/bleu2")
        bleu2_values = [name.split("_")[1] for name in imgs_names_bleu2]
        bleu2_values = [float(value[:-4]) for value in bleu2_values] 
        imgs_names_meteor = os.listdir(f"{root_dir}{folder_weights}/meteor")
        meteor_values = [name.split("_")[1] for name in imgs_names_meteor]
        meteor_values = [float(value[:-4]) for value in meteor_values] 
        plt.figure(figsize=(10, 10))
        metrics_values = np.array([bleu2_values, meteor_values])
        metrics_values = metrics_values.T
        
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_context("paper")
        # Add stripplot to boxplot
        
        # Add column names
        plt.xticks([0, 1], ["bleu2", "meteor"])
        sns.boxplot(data=metrics_values, showmeans=True, meanline=True)
        # plt.boxplot(metrics_values, labels=["bleu2", "meteor"], showmeans=True, meanline=True)
        # plt.ylabel(f"value")
        plt.ylim(0, 1)
        plt.savefig(f"{root_dir}{folder_weights}/boxplot_metrics.png")
        plt.close()
        metrics_values_sample = metrics_values[np.random.choice(metrics_values.shape[0], 400, replace=False), :]
        sns.stripplot(data=metrics_values_sample, jitter=True)
        plt.ylim(0, 1)
        plt.xticks([0, 1], ["bleu2", "meteor"])
        plt.savefig(f"{root_dir}{folder_weights}/stripplot_metrics.png")
        
        # Make the previous plots in the same plot, one at the left and the other at the right
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        # Make narrower the boxplot bars 
        
        sns.boxplot(data=metrics_values, width=0.6)
        plt.ylim(0, 1)
        plt.xticks([0, 1], ["bleu2", "meteor"])
        plt.subplot(1, 2, 2)
        sns.stripplot(data=metrics_values_sample, jitter=True)
        plt.ylim(0, 1)
        plt.xticks([0, 1], ["bleu2", "meteor"])
        plt.savefig(f"{root_dir}{folder_weights}/boxplot_stripplot_metrics.png")
        
        # Create a folder with the images with the best and worst metrics
        if not os.path.exists(f"{root_dir}{folder_weights}/best_images_bleu2"):
            os.makedirs(f"{root_dir}{folder_weights}/best_images_bleu2")
        if not os.path.exists(f"{root_dir}{folder_weights}/best_images_meteor"):
            os.makedirs(f"{root_dir}{folder_weights}/best_images_meteor")
        if not os.path.exists(f"{root_dir}{folder_weights}/worst_images_bleu2"):
            os.makedirs(f"{root_dir}{folder_weights}/worst_images_bleu2")
        if not os.path.exists(f"{root_dir}{folder_weights}/worst_images_meteor"):
            os.makedirs(f"{root_dir}{folder_weights}/worst_images_meteor")
            
        # Get the best and worst images
        imgs_best_bleu2 = sorted(imgs_names_bleu2, key=lambda x: float(x.split("_")[1][:-4]), reverse=True)[:num_imgs]
        imgs_best_meteor = sorted(imgs_names_meteor, key=lambda x: float(x.split("_")[1][:-4]), reverse=True)[:num_imgs]
        
        # Copy the best and worst images to the folder
        for img in imgs_best_bleu2:
            shutil.copy(f"{root_dir}{folder_weights}/bleu2/{img}", f"{root_dir}{folder_weights}/best_images_bleu2/{img}")
            
        for img in imgs_best_meteor:
            shutil.copy(f"{root_dir}{folder_weights}/meteor/{img}", f"{root_dir}{folder_weights}/best_images_meteor/{img}")
            
        imgs_worst_bleu2 = sorted(imgs_names_bleu2, key=lambda x: float(x.split("_")[1][:-4]))[:num_imgs]
        imgs_worst_meteor = sorted(imgs_names_meteor, key=lambda x: float(x.split("_")[1][:-4]))[:num_imgs]
        
        for img in imgs_worst_bleu2:
            shutil.copy(f"{root_dir}{folder_weights}/bleu2/{img}", f"{root_dir}{folder_weights}/worst_images_bleu2/{img}")
        
        for img in imgs_worst_meteor:
            shutil.copy(f"{root_dir}{folder_weights}/meteor/{img}", f"{root_dir}{folder_weights}/worst_images_meteor/{img}")
        
        

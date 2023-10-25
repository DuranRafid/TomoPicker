# Pumpkin: PU learning-based Macromolecule PicKINg in cryo-electron tomograms

# Import Statements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import scipy.stats as S
from sklearn.metrics import average_precision_score, matthews_corrcoef
import pandas as pd

import math
import sys
import os
import shutil
import mrcfile
import gc
import random

from abc import ABC, abstractmethod
from tqdm import tqdm
import argparse

# Method Architecture Implementation
class PumpkinEncoder(nn.Module):
    def __init__(self, subtomogram_size):
        super().__init__()
        self.encoder, self.latent_dim = self.__get_encoder(subtomogram_size)

    def __get_encoder(self, subtomogram_size):
        kernel_dims = None
        in_channels, out_channels, channel_scaling = 1, 32, 2
        layers = []

        if subtomogram_size == 16:
            kernel_dims = [4, 3, 3, 3]
        elif subtomogram_size == 32:
            kernel_dims = [8, 4, 4, 4, 4]

        layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dims[0], stride=2, bias=False)]
        layers += [nn.BatchNorm3d(num_features=out_channels)]
        layers += [nn.PReLU()]

        for kernel_dim in kernel_dims[1:]:
            in_channels = out_channels
            out_channels = out_channels * channel_scaling

            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, bias=False)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.PReLU()]

        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        return self.encoder(x)

class GaussianDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        assert 0 <= dropout < 1, "Invalid dropout provided!"

        self.dropout = dropout
        self.std = math.sqrt(dropout / (1 - dropout))

    def forward(self, x):
        if self.training and self.dropout > 0:
            means = torch.ones_like(input=x)
            gaussian_noises = torch.normal(mean=means, std=self.std)
            x = torch.mul(input=x, other=gaussian_noises)
        return x

class YOPOEncoder(nn.Module):
    def __init__(self, subtomogram_size):
        super().__init__()

        self.gaussian_dropout_unit = GaussianDropout(dropout=0.5)
        kernel_dim, self.latent_dim = None, None

        if subtomogram_size == 16:
            kernel_dim, self.latent_dim = 2, 256
        elif subtomogram_size == 32:
            kernel_dim, self.latent_dim = 3, 512

        self.convolution_units = nn.ModuleList()
        self.global_maxpool_units = nn.ModuleList()

        sum_out_channels, in_channels, out_channels, channel_increment = 0, 1, 64, 16

        for _ in range(10):
            self.convolution_units.append(self.__get_convolution_unit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim))
            self.global_maxpool_units.append(nn.AdaptiveMaxPool3d(output_size=1))

            sum_out_channels = sum_out_channels + out_channels
            in_channels = out_channels
            out_channels = out_channels + channel_increment

        self.batch_norm_unit = nn.BatchNorm3d(num_features=sum_out_channels, eps=0.001, momentum=0.99)
        self.downsample_unit = self.__get_downsample_unit(in_channels=sum_out_channels, downsample_dim=self.latent_dim)

    def __get_convolution_unit(self, in_channels, out_channels, kernel_size):
        layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)]
        layers += [nn.ELU()]
        layers += [nn.BatchNorm3d(num_features=out_channels, eps=0.001, momentum=0.99)]
        return nn.Sequential(*layers)

    def __get_downsample_unit(self, in_channels, downsample_dim):
        out_channels, channel_scaling = 1024, 2
        layers = []

        while not in_channels == downsample_dim:
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)]
            layers += [nn.ELU()]

            in_channels = out_channels
            out_channels = out_channels // channel_scaling

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gaussian_dropout_unit(x)

        global_maxpools = []

        for i in range(10):
            x = self.convolution_units[i](x)
            global_maxpools.append(self.global_maxpool_units[i](x))

        global_maxpool = torch.cat(tensors=global_maxpools, dim=1)
        batch_norm = self.batch_norm_unit(global_maxpool)
        return self.downsample_unit(batch_norm)

class PumpkinDecoder(nn.Module):
    def __init__(self, subtomogram_size, latent_dim, add_bias=True):
        super().__init__()
        self.decoder = self.__get_decoder(subtomogram_size, latent_dim, add_bias)

    def __get_decoder(self, subtomogram_size, latent_dim, add_bias):
        kernel_dims = [2, 2, 2]
        in_channels, out_channels, channel_scaling = latent_dim, latent_dim, 2
        layers = []

        for kernel_dim in kernel_dims:
            layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, bias=add_bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.LeakyReLU()]

            in_channels = out_channels
            out_channels = out_channels // channel_scaling

        if subtomogram_size == 16:
            kernel_dims = [4, 4]
        elif subtomogram_size == 32:
            kernel_dims = [4, 4, 4]

        for kernel_dim in kernel_dims[:-1]:
            layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dim, stride=2, padding=1, bias=add_bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.LeakyReLU()]

            in_channels = out_channels
            out_channels = out_channels // channel_scaling
        else:
            out_channels = 1

        layers += [nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_dims[-1], stride=2, padding=1, bias=add_bias)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class Pumpkin(nn.Module):
    def __init__(self, encoder_mode, subtomogram_size, use_decoder):
        super().__init__()

        if encoder_mode == "pumpkin":
            self.sample_encoder = PumpkinEncoder(subtomogram_size)
        elif encoder_mode == "yopo":
            self.sample_encoder = YOPOEncoder(subtomogram_size)

        self.sample_decoder = PumpkinDecoder(subtomogram_size, self.sample_encoder.latent_dim, add_bias=False) if use_decoder else None
        self.sample_classifier = PumpkinDecoder(subtomogram_size, self.sample_encoder.latent_dim)

    def forward(self, x):
        # Features extraction from subtomogram samples using encoder
        z = self.sample_encoder(x)
        # Sample reconstruction from extracted features using decoder
        y = self.sample_decoder(z) if self.sample_decoder is not None else None
        # Sample classification with extracted features using classifier
        x = self.sample_classifier(z)
        return x, y

# Training Objectives Implementation
class Objective(ABC):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.pi = pi
        self.recon_weight = reconstruction_weight
        self.l2_coefficient = l2_coefficient
        self.stats_headers = ["Loss", "Recon Error", "Precision", "TPR", "FPR"]
    
    def compute_score(self, features):
        features = torch.unsqueeze(features, dim=1)
        extracted_features = self.model.sample_encoder(features)
        score = self.model.sample_classifier(extracted_features).view(-1)
        return score
    
    def compute_recon_loss(self, features):
        features = torch.unsqueeze(features, dim=1)
        extracted_features = self.model.sample_encoder(features)
        recon_features = self.model.sample_decoder(extracted_features)
        recon_loss = (features - recon_features) ** 2
        recon_loss = torch.mean(torch.sum(recon_loss.view(recon_loss.size(0), -1), dim=1))
        return recon_loss
    
    def compute_performance_metrics(self, score, labels):
        p_hat = torch.sigmoid(score)
        precision = torch.sum(p_hat[labels == 1]).item() / torch.sum(p_hat).item()
        tpr = torch.mean(p_hat[labels == 1]).item()
        fpr = torch.mean(p_hat[labels == 0]).item()
        return precision, tpr, fpr
    
    def compute_regularization_loss(self):
        regularization_loss = sum([torch.sum((weights ** 2)) for weights in self.model.sample_encoder.parameters()])
        regularization_loss += sum([torch.sum((weights ** 2)) for weights in self.model.sample_classifier.parameters()])
        regularization_loss = 0.5 * self.l2_coefficient * regularization_loss
        return regularization_loss
    
    @abstractmethod
    def step(self, features, labels):
        pass

class PN(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)

    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)

        self.optimizer.zero_grad()

        if self.pi is not None:
            positive_loss = self.criteria(score[labels == 1], labels[labels == 1].float())
            negative_loss = self.criteria(score[labels == 0], labels[labels == 0].float())
            loss = positive_loss * self.pi + negative_loss * (1 - self.pi)
        else:
            loss = self.criteria(score, labels.float())

        if self.recon_weight > 0:
            loss = loss + recon_loss * self.recon_weight

        loss.backward()

        precision, tpr, fpr = self.compute_performance_metrics(score, labels)

        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()

        self.optimizer.step()

        return loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr

class PU(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)
        self.beta = 0
    
    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)

        self.optimizer.zero_grad()

        loss_pp = self.criteria(score[labels == 1], labels[labels == 1].float())
        loss_pn = self.criteria(score[labels == 1], 0 * labels[labels == 1].float())
        loss_un = self.criteria(score[labels == 0], labels[labels == 0].float())
        loss_u = loss_un - loss_pn * self.pi
        
        if loss_u.item() < -self.beta:
            loss = -loss_u
            backprop_loss = loss
            loss_u = -self.beta
            loss = loss_pp * self.pi + loss_u
        else:
            loss = loss_pp * self.pi + loss_u
            backprop_loss = loss
        
        if self.recon_weight > 0:
            backprop_loss = backprop_loss + recon_loss * self.recon_weight
        
        backprop_loss.backward()
        
        precision, tpr, fpr = self.compute_performance_metrics(score, labels)
        
        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()
        
        self.optimizer.step()
        
        return loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr

class GE_KL(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient, slack):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)
        self.slack = slack
        self.running_expectation = pi
        self.stats_headers += ["GE Penalty"]
        self.momentum = 1
        self.entropy_penalty = 0
    
    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None

        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)
        
        self.optimizer.zero_grad()
        
        classifier_loss = self.criteria(score[labels == 1], labels[labels == 1].float())
        p_hat = torch.mean(torch.sigmoid(score[labels == 0]))
        
        if self.momentum < 1:
            p_hat = p_hat * self.momentum + self.running_expectation * (1 - self.momentum)
            self.running_expectation = p_hat.item()
        
        entropy = np.log(self.pi) * self.pi + np.log1p(-self.pi) * (1 - self.pi)
        ge_penalty = -torch.log(p_hat) * self.pi - torch.log1p(-p_hat) * (1 - self.pi) + entropy
        ge_penalty = ge_penalty * self.slack / self.momentum
        
        loss = classifier_loss + ge_penalty
        
        if self.recon_weight > 0:
            loss = loss + recon_loss * self.recon_weight
        
        loss.backward()
        
        precision, tpr, fpr = self.compute_performance_metrics(score, labels)
        
        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()
        
        self.optimizer.step()
        
        return classifier_loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr, ge_penalty.item()

class GE_Binomial(Objective):
    def __init__(self, model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient, slack):
        super().__init__(model, criteria, optimizer, pi, reconstruction_weight, l2_coefficient)
        self.slack = slack
        self.stats_headers += ["GE Penalty"]
        self.entropy_penalty = 0
        self.posterior_l1 = 0
    
    def step(self, features, labels):
        score = self.compute_score(features)
        labels = labels.view(-1)
        recon_loss = None
        
        if self.recon_weight > 0:
            recon_loss = self.compute_recon_loss(features)
        
        self.optimizer.zero_grad()
        
        classifier_loss = self.criteria(score[labels == 1], labels[labels == 1].float())
        
        select = (labels.data == 0)
        N = torch.sum(select).item()
        p_hat = torch.sigmoid(score[select])
        q_mu = torch.sum(p_hat)
        q_var = torch.sum((p_hat * (1 - p_hat)))
        
        count_vector = torch.arange(start=0, end=(N + 1)).float()
        count_vector = count_vector.to(q_mu.device)
        
        q_discrete = -0.5 * (q_mu - count_vector) ** 2 / (q_var + sys.float_info.epsilon)
        q_discrete = F.softmax(q_discrete, dim=0)
        
        log_binom = S.binom.logpmf(k=np.arange(start=0, stop=(N + 1)), n=N, p=self.pi)
        log_binom = torch.from_numpy(log_binom).float()
        
        if q_var.is_cuda:
            log_binom = log_binom.cuda()
        
        ge_penalty = -torch.sum(log_binom * q_discrete)
        loss = classifier_loss + ge_penalty * self.slack
        
        if self.recon_weight > 0:
            loss = loss + recon_loss * self.recon_weight
        
        loss.backward()
        
        precision, tpr, fpr = self.compute_performance_metrics(score, labels)
        
        if self.l2_coefficient > 0:
            regularization_loss = self.compute_regularization_loss()
            regularization_loss.backward()
        
        self.optimizer.step()
        
        return classifier_loss.item(), recon_loss.item() if self.recon_weight > 0 else None, precision, tpr, fpr, ge_penalty.item()

# Utilities Implementation
def make_model(args):
    pumpkin = Pumpkin(encoder_mode=args.encoder_mode, subtomogram_size=args.subtomogram_size, use_decoder=args.use_decoder)
    pumpkin.train()
    return pumpkin

def load_model(args):
    pumpkin = Pumpkin(encoder_mode=args.encoder_mode, subtomogram_size=args.subtomogram_size, use_decoder=args.use_decoder)
    pumpkin.load_state_dict(torch.load(f=f"{args.model_path}/{args.model_name}.pt"))
    pumpkin.eval()
    return pumpkin

def load_tomogram(tomogram_path, clip_value=3):
    assert mrcfile.validate(tomogram_path)

    with mrcfile.open(tomogram_path) as mrc_file:
        tomogram = mrc_file.data

    tomogram = (tomogram - np.mean(tomogram)) / (np.std(tomogram) + sys.float_info.epsilon)
    tomogram = np.clip(tomogram, a_min=-clip_value - sys.float_info.epsilon, a_max=clip_value + sys.float_info.epsilon)
    tomogram = (tomogram - np.mean(tomogram)) / (np.std(tomogram) + sys.float_info.epsilon)
    return tomogram

def make_particles_mask(mask_shape, coords, particle_radius):
    particles_mask = np.zeros(shape=mask_shape, dtype=np.uint8)
    threshold = particle_radius ** 2

    z_grid = np.arange(start=0, stop=mask_shape[0], dtype=np.int32)
    y_grid = np.arange(start=0, stop=mask_shape[1], dtype=np.int32)
    x_grid = np.arange(start=0, stop=mask_shape[2], dtype=np.int32)
    z_grid, y_grid, x_grid = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')

    for i in range(coords.shape[0]):
        z_coord, y_coord, x_coord = coords[i, 0], coords[i, 1], coords[i, 2]
        squared_distances = (z_grid - z_coord) ** 2 + (y_grid - y_coord) ** 2 + (x_grid - x_coord) ** 2
        particles_mask = particles_mask + (squared_distances <= threshold).astype(np.uint8)

    return np.clip(particles_mask, a_min=0, a_max=1)

def make_data(args):
    tomogram_names, tomogram_paths = [], []

    for tomogram_file_name in os.listdir(args.tomograms_dir):
        tomogram_name, extension = os.path.splitext(tomogram_file_name)
        tomogram_names.append(tomogram_name)
        tomogram_paths.append(args.tomograms_dir + os.sep + tomogram_file_name)

    tomograms = pd.DataFrame({"tomogram_name": tomogram_names, "tomogram_path": tomogram_paths})
    coordinates = pd.read_csv(args.coordinates_path)

    if os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)

    subtomogram_names, total_regions, total_positive_regions = [], [], []

    for tomogram_name, tomogram_path in zip(tomograms.tomogram_name, tomograms.tomogram_path):
        # Loading tomogram from tomogram directory and preprocessing tomogram for downstream usage
        tomogram = load_tomogram(tomogram_path=tomogram_path)
        tomogram_shape = tomogram.shape

        # Selecting particles belonging to loaded tomogram
        coords = coordinates.loc[coordinates["tomogram_name"] == tomogram_name]

        # Filtering out particles located outside loaded tomogram
        z_min = y_min = x_min = args.subtomogram_size // 2
        z_max, y_max, x_max = [dim_shape - args.subtomogram_size // 2 for dim_shape in tomogram_shape]
        coords = coords[(z_min < coords.z_coord) & (coords.z_coord < z_max) & (y_min < coords.y_coord) & (coords.y_coord < y_max) & (x_min < coords.x_coord) & (coords.x_coord < x_max)]

        assert len(coords) > 0, f"No particles remaining for {tomogram_name}!"

        # Preparing particle and random coordinates before subtomograms and submasks generation
        particle_coords = coords[["z_coord", "y_coord", "x_coord"]].to_numpy(dtype=np.int32)
        random_coords = np.array([[random.randint(z_min, z_max), random.randint(y_min, y_max), random.randint(x_min, x_max)] for _ in range(int(args.random_subdata_percentage * particle_coords.shape[0]))], dtype=np.int32)
        coords = np.concatenate((particle_coords, random_coords), axis=0)

        # Generating subtomograms having particles located roughly at the centre
        subtomograms_path = args.data_dir + os.sep + "Subtomograms"

        if not os.path.exists(subtomograms_path):
            os.makedirs(subtomograms_path)

        for index in range(coords.shape[0]):
            subtomogram = tomogram[coords[index, 0] - args.subtomogram_size // 2:coords[index, 0] + args.subtomogram_size // 2, coords[index, 1] - args.subtomogram_size // 2:coords[index, 1] + args.subtomogram_size // 2, coords[index, 2] - args.subtomogram_size // 2:coords[index, 2] + args.subtomogram_size // 2]

            subtomogram_name = tomogram_name + "-subtomo-" + str(index)
            subtomogram_names.append(subtomogram_name)

            with open(f"{subtomograms_path}{os.sep}{subtomogram_name}.npy", 'wb') as npy_file:
                np.save(file=npy_file, arr=subtomogram)

        del tomogram
        gc.collect()

        # Generating particles mask of loaded tomogram as labels for downstream analyses
        particles_mask = make_particles_mask(mask_shape=tomogram_shape, coords=particle_coords, particle_radius=args.particle_radius)

        # Generating submasks having particles located roughly at the centre
        labels_path = args.data_dir + os.sep + "Labels"

        if not os.path.exists(labels_path):
            os.makedirs(labels_path)

        for index in range(coords.shape[0]):
            submask = particles_mask[coords[index, 0] - args.subtomogram_size // 2:coords[index, 0] + args.subtomogram_size // 2, coords[index, 1] - args.subtomogram_size // 2:coords[index, 1] + args.subtomogram_size // 2, coords[index, 2] - args.subtomogram_size // 2:coords[index, 2] + args.subtomogram_size // 2]

            submask_name = tomogram_name + "-labels-" + str(index)
            total_regions.append(submask.size)
            total_positive_regions.append(np.sum(submask))

            if np.sum(submask) == 0:
                submask = np.zeros(shape=submask.shape, dtype=np.uint8)
                submask[np.random.choice(submask.shape[0]), np.random.choice(submask.shape[1]), np.random.choice(submask.shape[2])] = 1

            with open(f"{labels_path}{os.sep}{submask_name}.npy", 'wb') as npy_file:
                np.save(file=npy_file, arr=submask)

        del particles_mask
        gc.collect()

    subtomogram_sizes, particle_radii = [args.subtomogram_size] * len(subtomogram_names), [args.particle_radius] * len(subtomogram_names)
    subtomograms_stats = pd.DataFrame({"subtomogram_name": subtomogram_names, "subtomogram_size": subtomogram_sizes, "particle_radius": particle_radii, "total_regions": total_regions, "total_positive_regions": total_positive_regions, "p_observed": [tpr / tr for tpr, tr in zip(total_positive_regions, total_regions)]})
    subtomograms_stats.to_csv(args.data_dir + os.sep + "subtomograms_stats.csv", index=False)

    return subtomograms_stats

def load_data(args):
    # Loading stats on subtomograms & labels and preparing subtomograms_stats for downstream usage
    subtomograms_stats = pd.read_csv(args.data_dir + os.sep + "subtomograms_stats.csv")
    subtomograms_stats = subtomograms_stats[["subtomogram_name", "total_regions", "total_positive_regions"]].values.tolist()
    subtomograms_stats = [[s_s[0].split('-')[0], s_s[0].split('-')[2], s_s[2], s_s[1] - s_s[2], "labeled"] for s_s in subtomograms_stats]

    # Shuffling subtomograms_stats for samples randomization and splitting it into training and testing subsets
    random.shuffle(subtomograms_stats)
    train_test_split_index = int(args.train_test_split_ratio * len(subtomograms_stats))
    train_samples_stats, test_samples_stats = subtomograms_stats[:train_test_split_index], subtomograms_stats[train_test_split_index:]

    # Shuffling train_samples_stats for samples randomization and splitting it into labeled and unlabeled subsets
    random.shuffle(train_samples_stats)
    labeled_unlabeled_split_index = int(args.particles_fraction * len(train_samples_stats))
    labeled_train_samples_stats = train_samples_stats[:labeled_unlabeled_split_index]
    unlabeled_train_samples_stats = train_samples_stats[labeled_unlabeled_split_index:]

    # Postprocessing unlabeled_train_samples_stats and determining class prior or fractions of positive regions in unlabeled samples
    unlabeled_train_samples_stats = [[s_s[0], s_s[1], s_s[2], s_s[3], "unlabeled"] for s_s in unlabeled_train_samples_stats]

    if args.particles_fraction == 1:
        args.objective_type = "PN" if args.objective_type in ["GE-KL", "GE-binomial"] else args.objective_type

    pi = None

    if args.objective_type in ["PN", "PU"]:
        total_positive_regions = sum([s_s[2] for s_s in labeled_train_samples_stats]) + sum([s_s[2] for s_s in unlabeled_train_samples_stats])
        total_negative_regions = sum([s_s[3] for s_s in labeled_train_samples_stats]) + sum([s_s[3] for s_s in unlabeled_train_samples_stats])
        positive_class_prior = total_positive_regions / (total_positive_regions + total_negative_regions)
        pi = positive_class_prior
    elif args.objective_type in ["GE-KL", "GE-binomial"]:
        total_unlabeled_positive_regions = sum([s_s[2] for s_s in unlabeled_train_samples_stats])
        total_negative_regions = sum([s_s[3] for s_s in labeled_train_samples_stats]) + sum([s_s[3] for s_s in unlabeled_train_samples_stats])
        positive_class_prior_in_unlabeled = total_unlabeled_positive_regions / (total_unlabeled_positive_regions + total_negative_regions)
        pi = positive_class_prior_in_unlabeled

    # Merging labeled_train_samples_stats and unlabeled_train_samples_stats into train_samples_stats
    train_samples_stats = labeled_train_samples_stats + unlabeled_train_samples_stats
    random.shuffle(train_samples_stats)

    return train_samples_stats, test_samples_stats, pi

def make_objective(args, model, pi):
    criteria = nn.BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=args.init_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    if args.objective_type == "PN":
        objective = PN(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient)
    elif args.objective_type == "PU":
        objective = PU(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient)
    elif args.objective_type == "GE-KL":
        objective = GE_KL(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient, slack=args.slack)
    elif args.objective_type == "GE-binomial":
        objective = GE_Binomial(model=model, criteria=criteria, optimizer=optimizer, pi=pi, reconstruction_weight=args.reconstruction_weight, l2_coefficient=args.l2_coefficient, slack=args.slack)
    else:
        objective = None

    return objective, criteria, lr_scheduler

class CustomSampleDataset(Dataset):
    def __init__(self, samples_stats, data_dir, augment_data):
        self.features, self.labels = [], []

        for sample_stats in samples_stats:
            self.features.append(sample_stats[0] + "-subtomo-" + sample_stats[1])
            self.labels.append([sample_stats[0] + "-labels-" + sample_stats[1], sample_stats[4]])

        self.data_dir, self.augment_data = data_dir, augment_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        with open(f"{self.data_dir}{os.sep}Subtomograms{os.sep}{self.features[idx]}.npy", 'rb') as npy_file:
            subtomogram = np.load(file=npy_file)

        if self.labels[idx][1] == "labeled":
            with open(f"{self.data_dir}{os.sep}Labels{os.sep}{self.labels[idx][0]}.npy", 'rb') as npy_file:
                submask = np.load(file=npy_file)
        elif self.labels[idx][1] == "unlabeled":
            submask = np.zeros(shape=subtomogram.shape, dtype=np.uint8)
            submask[np.random.choice(submask.shape[0]), np.random.choice(submask.shape[1]), np.random.choice(submask.shape[2])] = 1
        else:
            submask = None

        if self.augment_data:
            choice_value = np.random.uniform()

            if choice_value < 0.3:
                num_rotations = np.random.randint(low=1, high=5)
                rotation_axes = np.random.choice(a=[0, 1, 2], size=(2,), replace=False)
                subtomogram = np.rot90(m=subtomogram, k=num_rotations, axes=rotation_axes).copy()
                submask = np.rot90(m=submask, k=num_rotations, axes=rotation_axes).copy()
            elif choice_value < 0.6:
                flip_axis = np.random.choice(a=[0, 1, 2])
                subtomogram = np.flip(m=subtomogram, axis=flip_axis).copy()
                submask = np.flip(m=submask, axis=flip_axis).copy()

        return subtomogram, submask

class CustomBatchSampler(BatchSampler):
    def __init__(self, num_samples, num_batches, batch_size):
        self.sample_indices = list(range(num_samples))
        self.num_batches = num_batches
        self.batch_size = batch_size
        super().__init__(sampler=self.sample_indices, batch_size=self.batch_size, drop_last=False)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield random.sample(self.sample_indices, k=self.batch_size)

def make_train_dataloader(args, samples_stats):
    dataset = CustomSampleDataset(samples_stats=samples_stats, data_dir=args.data_dir, augment_data=args.augment_data)

    if args.augment_data:
        batch_sampler = CustomBatchSampler(num_samples=len(dataset), num_batches=args.num_iterations, batch_size=args.train_batch_size)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

    return dataloader

def make_test_dataloader(args, samples_stats):
    dataset = CustomSampleDataset(samples_stats=samples_stats, data_dir=args.data_dir, augment_data=False)
    dataloader = DataLoader(dataset=dataset, batch_size=args.test_batch_size)
    return dataloader

def fit_epoch(epoch, objective, train_dataloader, output):
    for iteration, (features, labels) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()

        metrics = objective.step(features, labels)
        output += '\n' + '\t'.join([str(epoch + 1), str(iteration + 1), "train"] + [str(metric) if metric is None else f"{metric:.5f}" for metric in metrics] + ['-'] * 2)
    return output

def test_model(model, criteria, test_dataloader):
    if model.training:
        model.eval()

    num_sample_points = loss = 0
    y_score, y_true = [], []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = torch.unsqueeze(features, dim=1)
            labels = labels.view(-1)
            y_true.append(labels.numpy())

            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()

            score, _ = model(features)
            score = score.view(-1)

            y_score.append(score.cpu().numpy())
            running_loss = criteria(score, labels.float()).item()
            num_sample_points = num_sample_points + labels.size(0)
            delta = labels.size(0) * (running_loss - loss)
            loss = loss + delta / num_sample_points

    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_hat = 1 / (1 + np.exp(-y_score))
    y_predicted = (y_hat >= 0.5).astype(y_true.dtype)

    precision = np.sum(y_hat[y_true == 1]) / np.sum(y_hat)
    tpr = np.mean(y_hat[y_true == 1])
    fpr = np.mean(y_hat[y_true == 0])
    auprc = average_precision_score(y_true=y_true, y_score=y_hat)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_predicted)

    return loss, precision, tpr, fpr, auprc, mcc

def train_test_model(args, model, objective, criteria, lr_scheduler, train_dataloader, test_dataloader):
    stats_headers, args_dict = '\t'.join(["Epoch", "Iteration", "Split"] + objective.stats_headers + ["AUPRC", "MCC"]), vars(args)
    stats = "-- Pumpkin Training & Testing --\n\n" + '\n'.join([f"{key}: {args_dict[key]}" for key in args_dict]) + '\n'

    max_test_auprc = -np.inf

    for epoch in tqdm(iterable=range(args.num_epochs), desc="Training Progress", ncols=100, unit="epoch"):
        if not model.training:
            model.train()

        stats += '\n' + stats_headers

        stats = fit_epoch(epoch=epoch, objective=objective, train_dataloader=train_dataloader, output=stats)
        loss, precision, tpr, fpr, auprc, mcc = test_model(model=model, criteria=criteria, test_dataloader=test_dataloader)

        test_stats = '\t'.join([str(epoch + 1), '-', "test", f"{loss:.5f}", '-', f"{precision:.5f}", f"{tpr:.5f}", f"{fpr:.5f}"] + (['-'] if args.objective_type in ["GE-KL", "GE-binomial"] else []) + [f"{auprc:.5f}", f"{mcc:.5f}"])
        stats += "\n\n" + stats_headers + '\n' + test_stats + '\n'

        if max_test_auprc < auprc:
            stats += f"\nTest AUPRC score improved from {max_test_auprc:.5f} to {auprc:.5f}.\n"
            max_test_auprc = auprc

            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)

            torch.save(obj=model.state_dict(), f=f"{args.model_path}{os.sep}{args.model_name}.pt")
            stats += f"Updated model {args.model_name}.pt saved.\n"
        else:
            stats += f"\nTest AUPRC score did not improve from {max_test_auprc:.5f}.\n"

        lr_scheduler.step(max_test_auprc)
        stats += f"Current learning rate is {lr_scheduler._last_lr[0]}.\n"

    stats += "\nDone!"

    if not os.path.exists(args.stats_path):
        os.makedirs(args.stats_path)

    with open(args.stats_path + os.sep + args.model_name + "_train_test_stats.txt", 'w') as stats_file:
        stats_file.write(stats)

    return stats

def get_args():
    parser = argparse.ArgumentParser(description="Python script for training a Pumpkin model to pick macromolecules from cryo-electron tomograms.")
    metavar = 'X'

    parser.add_argument("--input", default=None, type=str, metavar=metavar, help="Path to a folder containing input subtomograms and submasks (Default: None)", dest="data_dir")
    parser.add_argument("--tomogram", default=None, type=str, metavar=metavar, help="Path to a folder containing sample tomograms for data generation (Default: None)", dest="tomograms_dir")
    parser.add_argument("--coord", default=None, type=str, metavar=metavar, help="Path to the particle coordinates file for data generation (Default: None)", dest="coordinates_path")

    parser.add_argument("--encoder", default="pumpkin", type=str, metavar=metavar, help="Type of feature extractor (either pumpkin or yopo) to use in network (Default: pumpkin)", dest="encoder_mode")
    parser.add_argument("--decoder", action="store_true", help="Whether to use sample reconstructor in network (Default: False)", dest="use_decoder")

    parser.add_argument("--size", default=16, type=int, metavar=metavar, help="Size of subtomograms and submasks (either 16 or 32) in each dimension (Default: 16)", dest="subtomogram_size")
    parser.add_argument("--radius", default=7, type=int, metavar=metavar, help="Radius of a particle (in pixel) in sample tomograms (Default: 7)", dest="particle_radius")
    parser.add_argument("--random", default=0.25, type=float, metavar=metavar, help="Percentage of randomly generated subtomograms and submasks (Default: 0.25)", dest="random_subdata_percentage")

    parser.add_argument("--split", default=0.8, type=float, metavar=metavar, help="Percentage of samples used in training (Default: 0.8)", dest="train_test_split_ratio")
    parser.add_argument("--fraction", default=0.1, type=float, metavar=metavar, help="Percentage of samples with proper labeling for training (Default: 0.1)", dest="particles_fraction")

    parser.add_argument("--epoch", default=10, type=int, metavar=metavar, help="Number of training epochs (Default: 10)", dest="num_epochs")
    parser.add_argument("--iter", default=100, type=int, metavar=metavar, help="Number of weight updates in each training epoch (Default: 100)", dest="num_iterations")
    parser.add_argument("--train_batch", default=256, type=int, metavar=metavar, help="Number of samples in a training batch (Default: 256)", dest="train_batch_size")
    parser.add_argument("--test_batch", default=1, type=int, metavar=metavar, help="Number of samples in a testing batch (Default: 1)", dest="test_batch_size")

    parser.add_argument("--init_lr", default=2e-5, type=float, metavar=metavar, help="Initial lr used in training (Default: 2e-5)", dest="init_lr")
    parser.add_argument("--factor", default=0.5, type=float, metavar=metavar, help="Factor used in lr scheduling to adjust lr (Default: 0.5)", dest="factor")
    parser.add_argument("--patience", default=0, type=int, metavar=metavar, help="Patience used in lr scheduling to adjust lr (Default: 0)", dest="patience")
    parser.add_argument("--min_lr", default=1e-7, type=float, metavar=metavar, help="Minimum lr allowed in training (Default: 1e-7)", dest="min_lr")

    parser.add_argument("--name", default="pumpkin", type=str, metavar=metavar, help="Name of the model (Default: pumpkin)", dest="model_name")
    parser.add_argument("--save_weight", default=None, type=str, metavar=metavar, help="Path to a folder for saving model weights (Default: None)", dest="model_path")
    parser.add_argument("--save_stat", default=None, type=str, metavar=metavar, help="Path to a folder for saving training and testing stats (Default: None)", dest="stats_path")

    parser.add_argument("--recon_weight", default=0.0, type=float, metavar=metavar, help="Weight on sample reconstruction error in loss calculation (Default: 0.0)", dest="reconstruction_weight")
    parser.add_argument("--l2_coeff", default=0.0, type=float, metavar=metavar, help="Weight on L2 regularization term (Default: 0.0)", dest="l2_coefficient")

    parser.add_argument("--objective", default="PU", type=str, metavar=metavar, help="Type of objective (any one of PN, PU, GE-KL or GE-binomial) to use in training (Default: PU)", dest="objective_type")
    parser.add_argument("--slack", default=None, type=float, metavar=metavar, help="Value of slack to use in GE-KL or GE-binomial objective (Default: None)", dest="slack")

    parser.add_argument("--make", action="store_true", help="Whether to generate input dataset before training (Default: False)", dest="make_dataset")
    parser.add_argument("--augment", action="store_true", help="Whether to augment input dataset during training (Default: False)", dest="augment_data")

    parser.set_defaults(use_decoder=False)
    parser.set_defaults(make_dataset=False)
    parser.set_defaults(augment_data=False)

    args = parser.parse_args()

    encoder_modes = ["pumpkin", "yopo"]
    objective_types = ["PN", "PU", "GE-KL", "GE-binomial"]
    slack_configs = {"GE-KL": 10, "GE-binomial": 1}

    assert args.encoder_mode in encoder_modes, "Invalid encoder_mode provided!"
    args.use_decoder = args.reconstruction_weight > 0

    assert args.subtomogram_size == 16 or args.subtomogram_size == 32, "Invalid subtomogram_size provided!"
    assert args.particle_radius > 0, "Invalid particle_radius provided!"
    assert args.random_subdata_percentage >= 0, "Invalid random_subdata_percentage provided!"

    assert 0 < args.train_test_split_ratio < 1, "Invalid train_test_split_ratio provided!"
    assert 0 < args.particles_fraction <= 1, "Invalid particles_fraction provided!"

    assert args.reconstruction_weight >= 0, "Invalid reconstruction_weight provided!"
    assert args.l2_coefficient >= 0, "Invalid l2_coefficient provided!"

    assert args.objective_type in objective_types, "Invalid objective_type provided!"
    args.slack = slack_configs[args.objective_type] if args.objective_type in slack_configs else None

    return args

if __name__ == "__main__":
    # Processing command line arguments
    args = get_args()

    # Producing sample subtomograms and submasks (if necessary)
    if args.make_dataset:
        make_data(args=args)

    # Making an instance of Pumpkin class for training and testing
    pumpkin = make_model(args=args)

    if torch.cuda.is_available():
        pumpkin = pumpkin.cuda()

    # Loading samples_stats and fraction of positive regions in unlabeled samples (pi)
    train_samples_stats, test_samples_stats, pi = load_data(args=args)
    print(f"\n#Train Samples = {len(train_samples_stats)}, #Test Samples = {len(test_samples_stats)}, Pi = {pi}\n")

    # Making objective function for training pipeline
    objective, criteria, lr_scheduler = make_objective(args=args, model=pumpkin, pi=pi)

    # Making training and testing dataloaders for training pipeline
    train_dataloader = make_train_dataloader(args=args, samples_stats=train_samples_stats)
    test_dataloader = make_test_dataloader(args=args, samples_stats=test_samples_stats)

    # Training and testing a model for picking particles from tomograms
    train_test_model(args=args, model=pumpkin, objective=objective, criteria=criteria, lr_scheduler=lr_scheduler, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
import argparse
import pandas as pd
import numpy as np
import os
import torch
from train_pumpkin import load_tomogram, load_model
from torch_disca import main_fn

def generate_subtomograms_and_coords(tomogram, subtom_dim):
    z, x, y = tomogram.shape
    stride = subtom_dim

    subtomograms = []
    starting_coords = []

    for i in range(0, z, stride):
        for j in range(0, x, stride):
            for k in range(0, y, stride):
                # get the subtomogram
                # if the subtomogram is not of shape subtom_dim x subtom_dim x subtom_dim, then pad it with zeros
                if i+subtom_dim > z or j+subtom_dim > x or k+subtom_dim > y:
                    subtomogram = np.zeros((subtom_dim, subtom_dim, subtom_dim))
                    tmp = tomogram[i:i+subtom_dim, j:j+subtom_dim, k:k+subtom_dim]
                    subtomogram[:tmp.shape[0], :tmp.shape[1], :tmp.shape[2]] = tmp
                else:
                    subtomogram = tomogram[i:i+subtom_dim, j:j+subtom_dim, k:k+subtom_dim]
                # store the subtomogram with starting coordinates
                starting_coord = (i, j, k)
                subtomograms.append(subtomogram)
                starting_coords.append(starting_coord)
    return subtomograms, starting_coords

def merge_score_matrices(score, starting_coords, subtom_dim, tom_dim):
    z, x, y = tom_dim
    output = np.zeros(tom_dim, dtype=np.float32)
    # loop over the score matrices
    for i in range(score.shape[0]):
        score_matrix = score[i]
        starting_coord = starting_coords[i]
        i, j, k = starting_coord
        # if the score matrix is not of shape subtom_dim x subtom_dim x subtom_dim, then pad it with zeros
        if i+subtom_dim > z or j+subtom_dim > x or k+subtom_dim > y:
            tmp = np.zeros((subtom_dim, subtom_dim, subtom_dim))
            tmp[:score_matrix.shape[0], :score_matrix.shape[1], :score_matrix.shape[2]] = score_matrix
            score_matrix = tmp
        # add the score matrix to the output
        if i+subtom_dim > z or j+subtom_dim > x or k+subtom_dim > y:
            output[i:i+subtom_dim, j:j+subtom_dim, k:k+subtom_dim] += score_matrix[:z-i, :x-j, :y-k]
        else:
            output[i:i+subtom_dim, j:j+subtom_dim, k:k+subtom_dim] += score_matrix
    return output

def nms3d_optimized_halka_patla(scores_matrix, radius, num_expected_particles=100, threshold=0):
    z_max, x_max, y_max = scores_matrix.shape
    estimations = []

    for _ in range(num_expected_particles):
        max_score_position = np.unravel_index(indices=np.argmax(scores_matrix), shape=scores_matrix.shape)

        if scores_matrix[max_score_position] <= threshold:
            break

        z, x, y = max_score_position
        score = scores_matrix[max_score_position]

        scores_matrix[
            max(0, z - radius):min(z_max, z + radius),
            max(0, x - radius):min(x_max, x + radius),
            max(0, y - radius):min(y_max, y + radius)
        ] = -np.inf
        estimations.append([x, y, z, score])

    return estimations

def generate_estimations(tomogram_name, scores_matrix, radius, num_expected_particles=100):
    estimations = nms3d_optimized_halka_patla(scores_matrix, radius, num_expected_particles)
    estimations_dataframe = pd.DataFrame(
        {
            "tomogram_name": [tomogram_name] * len(estimations),
            "x_coord": [item[0] for item in estimations],
            "y_coord": [item[1] for item in estimations],
            "z_coord": [item[2] for item in estimations],
            "score": [item[3] for item in estimations]
        }
    )
    return estimations, estimations_dataframe

def generate_subtomograms_for_clustering(estimations, radius, tomogram):
    subtomograms = []
    for estimation in estimations:
        x, y, z = estimation[0], estimation[1], estimation[2]

        subtomogram = np.zeros((radius * 2 + 1, radius * 2 + 1, radius * 2 + 1))
        temp_tomogram = tomogram[
            max(0, x - radius):min(tomogram.shape[0], x + radius),
            max(0, y - radius):min(tomogram.shape[1], y + radius),
            max(0, z - radius):min(tomogram.shape[2], z + radius)
        ]
        subtomogram[
            :temp_tomogram.shape[0],
            :temp_tomogram.shape[1],
            :temp_tomogram.shape[2]
        ] = temp_tomogram

        subtomograms.append(subtomogram)

    subtomograms = np.array(subtomograms, dtype=np.float32)
    return subtomograms

def get_args():
    parser = argparse.ArgumentParser(description="Python script for a Pumpkin model inference pipeline to pick macromolecules from cryo-electron tomograms.")
    metavar = 'X'

    parser.add_argument("--input", default=None, type=str, metavar=metavar, help="Path to a folder containing input subtomograms and submasks (Default: None)", dest="data_dir")
    parser.add_argument("--tomogram", default=None, type=str, metavar=metavar, help="Path to a folder containing sample tomograms for data generation (Default: None)", dest="tomograms_dir")

    parser.add_argument("--tomograms_path", default="./ribo10064_bin8/MRCs", type=str, metavar=metavar, help="Path to a folder containing sample tomograms for data generation (Default: ./ribo10064_bin8/MRCs)", dest="tomograms_path")
    parser.add_argument("--tomogram_name", default=None, type=str, metavar=metavar, help="Name of the tomogram to be used for data generation (Default: None)", dest="tomogram_name")

    parser.add_argument("--encoder", default="topaz", type=str, metavar=metavar, help="Type of feature extractor (topaz or yopo) to use in network (Default: topaz)", dest="encoder_mode")
    parser.add_argument("--size", default=16, type=int, metavar=metavar, help="Size of subtomograms and submasks in each dimension (Default: 16)", dest="subtomogram_size")
    parser.add_argument("--radius", default=7, type=int, metavar=metavar, help="Radius of a particle (in pixel) in sample tomograms (Default: 7)", dest="particle_radius")

    parser.add_argument("--name", default=None, type=str, metavar=metavar, help="Name of the model (Default: pumpkin)", dest="model_name")
    parser.add_argument("--model_path", default="./results/models", type=str, metavar=metavar, help="Path to a folder for saving model weights (Default: None)", dest="model_path")
    parser.add_argument("--should_cluster", default=False, type=bool, metavar=metavar, help="Whether to perform clustering (Default: False)", dest="should_cluster")
    parser.add_argument("--num_expected_particles", default=100, type=int, metavar=metavar, help="Number of expected particles (Default: 100)", dest="num_expected_particles")
    parser.set_defaults(use_decoder=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    tomogram = load_tomogram(f"{args.tomograms_path}/{args.tomogram_name}.mrc")
    tomogram_shape = tomogram.shape

    subtomograms, starting_coords = generate_subtomograms_and_coords(
        tomogram = tomogram,
        subtom_dim = args.subtomogram_size
    )
    print("Number of subtomograms:", len(subtomograms))

    subtomograms = np.array(subtomograms, dtype=np.float32)
    subtomograms = torch.from_numpy(subtomograms)
    subtomograms = torch.unsqueeze(subtomograms, dim=1)

    model_infer = load_model(args)
    scores, _ = model_infer(subtomograms)

    scores_mesh = np.reshape(scores.detach().numpy(), newshape=(scores.shape[0],) + scores.shape[2:])
    scores_matrix = merge_score_matrices(
        scores_mesh,
        starting_coords,
        args.subtomogram_size,
        tomogram_shape
    )

    estimations, estimations_dataframe = generate_estimations(
        args.tomogram_name,
        scores_matrix,
        args.particle_radius,
        args.num_expected_particles
    )
    print("Number of estimations:", len(estimations))

    ## Save
    picked_particle_path = f'results/particle/{args.model_name}'
    if not os.path.exists(picked_particle_path):
        os.makedirs(picked_particle_path)
    estimations_dataframe.to_csv(picked_particle_path + os.sep + "estimations_" + args.tomogram_name + f"_{args.num_expected_particles}" + ".csv")

    if args.should_cluster:

        subtomograms_for_clustering = generate_subtomograms_for_clustering(estimations, args.particle_radius, tomogram)

        dir_to_save = f"./results/subtomograms_for_clustering/{args.model_name}"
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)

        np.save(f"./results/subtomograms_for_clustering/{args.model_name}/{args.tomogram_name}.npy", subtomograms_for_clustering)
        print(f"Subtomograms for clustering saved to {dir_to_save}/{args.tomogram_name}.npy")

        main_fn(args.model_name, args.tomogram_name)

        clustering_path = f'results/clustering/disca/{args.model_name}'
        label_path = f'results/clustering/disca/{args.model_name}/labels_torch_{args.tomogram_name}.pickle'
        cluster_labels = np.load(label_path, allow_pickle=True)

        estimations_dataframe["cluster"] = cluster_labels
        estimations_dataframe_zero = estimations_dataframe.loc[estimations_dataframe["cluster"] == 0]
        estimations_dataframe_one = estimations_dataframe.loc[estimations_dataframe["cluster"] == 1]

        estimations_dataframe_zero.to_csv(clustering_path + os.sep + "estimations_0_" + args.tomogram_name + ".csv", index=False)
        estimations_dataframe_one.to_csv(clustering_path + os.sep + "estimations_1_"  + args.tomogram_name + ".csv", index=False)

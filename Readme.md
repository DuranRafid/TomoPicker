## Installation

First, create a conda virtual environment.

``````sh
conda create -n tomopicker -c conda-forge python=3.8.3 -y 
``````

Activate the environment:

``````sh
conda activate tomopicker
``````

Install the necessary packages in the environment:

``````sh
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y

pip install -r requirement.txt
``````

## Training 

For training the model, you need to run `train_tomopicker.py` using python. You may run this Python script to train a PU learning-based model for picking macromolecular particles from tomograms. In the following sections, you will find some examples of running this script and some notes on command-line arguments that this script accepts.

### Examples

-   To see details on command-line arguments.
    ``````sh
    python train_tomopicker.py -h or python train_tomopicker.py --help
    ``````

-   To run the script for creating a new input dataset and then training a model.
    ``````sh
    python train_tomopicker.py  --tomogram <tomogram_path> --coord <annotated_coordinate_path> --name <model_name> --save_weight <path_for_saved_model> --save_log <path_for_saved_train_log> --objective <objective_type> --encoder <Network_type> --pick <expected_number_of_particles_in_the_training_tomogram> --radius <particle_radius> --size <subtomogram_size> --make
    ``````
    
-   To run the script for training a model.
    
    ``````sh
    python train_tomopicker.py --input <input_data_path> --name <model_name> --save_weight <path_for_saved_model> --save_stat <path_for_saved_result> --objective <objective_type> --encoder <Network_type> --pick <expected_number_of_particles_in_the_training_tomogram> --radius <particle_radius> --size <subtomogram_size>
    ``````

Three objective types are supported: PN, PU, and GE-KL. Two network types are supported: basic and unet. However, unet always give superior results. In our experiments, we use only one tomogram for training and one for validation. 

## Inference 

For training the model, you need to run `inference_tomopicker.py` using python. You may run this Python script to perform picking from any tomogram using a saved model. In the following sections, you will find some examples of running this script and some notes on command-line arguments that this script accepts. 

### Examples
    
-   To run the script for picking particles in a tomogram given a saved model.
    
    ``````sh
    python inference_pumpkin.py --tomograms_path <path_to_tomograms> --tomogram_name <name_of_the_tomogram> --encoder <encoder_type> --name <saved_model_name> --model_path <path_for_saved_model> --pick <expected_number_of_particles_from_the_tomogram> --radius <particle_radius> --size <subtomogram_size>
    ``````

This code saves the estimated particle coordinates with probability scores under the folder ``estimations/<model_name>``


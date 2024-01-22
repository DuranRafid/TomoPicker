## How to Run `train_pumpkin.py`

You may run this Python script to train a PU learning-based model for picking macromolecular particles from tomograms. In the following sections, you will find some examples of running this script and some notes on command-line arguments that this script accepts.

### Examples

-   To see details on command-line arguments.
    ``````sh
    python train_pumpkin.py -h or python train_pumpkin.py --help
    ``````

-   To run the script for creating a new input dataset and then training a model.
    ``````sh
    python train_pumpkin.py --input ../ribo10064_bin8/data --tomogram ../ribo10064_bin8/MRCs --coord ../ribo10064_bin8/documents/empiar10064_bin8.csv --name experimental --save_weight ../results/models --save_stat ../results/stats --make --augment
    ``````
    
-   To run the script for training a model.
    
    ``````sh
    python train_pumpkin.py --input ../ribo10064_bin8/data --name experimental --save_weight ../results/models --save_stat ../results/stats --augment
    ``````

### Command-line Arguments

-   You will find in **[this folder](https://drive.google.com/drive/folders/1WrUaTdpGKQrwMtrGG9vv_xHYFs15fe-7?usp=sharing)** necessary input subtomograms (feature) as well as submasks (label) for training the model. Besides, you will find here the sample tomograms from EMPIAR-10064 dataset and a `.csv` file containing particle coordinates belonging to these tomograms. ***You may need the latter for new input dataset creation by providing `--make` argument.***
-   Value of command-line argument `--encoder` can by any one of `pumpkin` and `yopo`.
-   Value of command-line argument `--decoder` will be set to `True` inside the script when the value of another argument `--recon_weight` is greater than `0`.
-   Value of command-line argument `--size` must be either `16` or `32`.
-   Value of command-line argument `--radius` denotes the number of pixels around a particle center to label as positive (`1`).
-   Value of command-line argument `--random` represents the percentage***, with respect to total number of particles in a sample tomogram,*** of randomly generated input subtomograms/submasks along with original input dataset.
-   Value of command-line argument `--split` denotes the fraction of input subtomograms/submasks to retain for training.
-   Value of command-line argument `--fraction` represents the fraction of training subtomograms/submasks to retain as labeled for training a model in PU learning setting.
-   Value of command-line argument `--objective` can be any one of `PN`, `PU`, `GE-KL`, and `GE-binomial`.
-   Existing input dataset (subtomograms and submasks) is deleted, and a new dataset is created according to `--size`, `--radius`, and `--random` arguments when the command-line argument `--make` is provided. ***You may want to create a new input dataset when running this script for the first time.***
-   Input data augmentation takes place when the command-line argument `--augment` is provided.
# Deep Learning Models for Eye Tracking Trials

## Installation

We recommend using a conda environment with ``Python >= 3.12`` :

```bash
conda create -n precog python=3.12
conda activate precog
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/usc-sail/precog-eye-dl
cd precog-eye-dl
pip install -r requirements.txt
```

You should have access to the data folder named ``eyelink_processed``. You should copy this folder in this directory.

## Code Structure

* ``config.yaml`` contains sample values for the experiment variables.
* ``dataset.py`` contains the dataset class for the eye tracking data.
* ``model.py`` contains the model classes to be used for modeling (adapted from TimesNet).
* ``trainer.py`` ontains the class to perform train and evaluation.
* ``script.py`` is the main script to run for model traning.
* ``preprocess_*.py`` creates the input data for the model from ASCII files.

## Data Structure

The data for this code is located under ``/PATH/TO/eyelink-processed/``.

## Pre-processing

You do not need to run pre-processing, since all the data have been processed and are available in the data folder ``input_30trials``. To reproduce this sub-folder you need to run the ``preprocess.py`` and ``preprocess_1_7.py`` (for first 7 participants) scripts. These scripts read the raw ``asc_files``.

## Running the models

To run the deep learning pipeline for C v DS training (adjust for C v S inside the code), you should run

```bash
python script.py
```

Most parameters can be tuned withing the ``config.yaml`` file.

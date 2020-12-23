# Concept Detection in Medical images using Xception model
In this repository, we implemented our paper(http://ceur-ws.org/Vol-2696/paper_109.pdf) used to solve the ImageClef Concept dection task 2020. 

## Table of contents
* [Task Description](#Task_Description)
* [Installation](#Installation)
* [Usage](#Usage)
## Task Description
The dataset consist of 56,629 Training images and 14159 validation images.
The test dataset consists of 10,000 images.
The task was to predict concepts in medical Images
```bash
ROCO_CLEF_41341 C0033785;C0035561
```
## Installation

Use the package manager [conda](https://anaconda.org/anaconda/conda) to install requirements.

```bash
conda env create --file requirements.yml
```

## Usage
To train this project:
```
$  python training.py -tr Training-Concepts.txt -vl Validation-Concepts.txt -c stringconcepts.csv --batch_size 32 -ep 1 -lr 1e-5 -imgz 150
```
To test the model on dataset:
```
python testing.py -tr Training-Concepts.txt -c stringconcepts.csv --batch_size 1
```
To evaluate the model on the dataset:
```
python evaluate-f1.py /path/to/candidate/file /path/to/ground-truth/file
```
To calculate accuracy between predicted result and ground truth:
```
python calculate_accuracy.py -r results.csv -gt groundtruth.csv
```




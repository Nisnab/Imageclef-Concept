# Concept Detection in Medical images using Xception model
In this repository, we implemented our paper(http://ceur-ws.org/Vol-2696/paper_109.pdf) used to solve the ImageClef Concept dection task 2020. 

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
$ python training.py 
```


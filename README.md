# Automated-Cardiac-Segmentation-and-Disease-Diagnosis
## Introduction
This repository contains the implementation for automated cardiac segmentation and diasease classification introduced in the following paper: "Fully Convolutional Multi-scale Residual DenseNets for Cardiac Segmentation and Automated Cardiac Diagnosis using Ensemble of Classifiers" https://doi.org/10.1016/j.media.2018.10.004

### Citation
If you find this implementation useful in your research, please consider citing:

```
@article{khened2019fully,
  title={Fully convolutional multi-scale residual DenseNets for cardiac segmentation and automated cardiac diagnosis using ensemble of classifiers},
  author={Khened, Mahendra and Kollerathu, Varghese Alex and Krishnamurthi, Ganapathy},
  journal={Medical image analysis},
  volume={51},
  pages={21--45},
  year={2019},
  publisher={Elsevier}
}
```

## Usage

### ACDC Data Preparation
1. Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
2. Create a folder outside the project with name **ACDC_DataSet** and copy the dataset.
3. From the project folder open file data_preprocess/acdc_data_preparation.py
4. In the file, set the path to ACDC training dataset is pointed as: ```complete_data_path = '../../ACDC_DataSet/training' ```
5. Run the script acdc_data_preparation.py

### Steps to train the model:


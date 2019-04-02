# Automated-Cardiac-Segmentation-and-Disease-Diagnosis
## Introduction
This repository contains the reference implementation for automated cardiac segmentation and diasease classification introduced in the following paper: "Fully Convolutional Multi-scale Residual DenseNets for Cardiac Segmentation and Automated Cardiac Diagnosis using Ensemble of Classifiers" https://doi.org/10.1016/j.media.2018.10.004

### Citation
If you find this reference implementation useful in your research, please consider citing:

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
3. From the project folder open file data_preprocess/acdc_data_preparation.py.
4. In the file, set the path to ACDC training dataset is pointed as: ```complete_data_path = '../../ACDC_DataSet/training' ```.
5. Run the script acdc_data_preparation.py.
6. The processed data for training is generated outside the project folder named *processed_acdc_dataset*.

### Steps to train the model:
1. From the project folder open file estimators/train.py and configure the network hyper-parameters.
2. From the project folder open file estimators/config.py and configure the training hyper-parameters.
3. Run the script train.py.
4. Outside the project in the folder named *trained_models/ACDC/* the model weights and tensorboard summary are saved.
5. While training the training summary can be accessed running: ```tensorboard --logdir='path_to/trained_models/ACDC/FCRD_ACDC/summary' ```.

### Steps to test the model:
1. From the project folder open file estimators/test.py and configure the testing hyper-parameters, path to trained model weights and ACDC-2017 testing dataset.
2. Run the script test.py.
3. The predictions are saved in the path *trained_models/ACDC/FCRD_ACDC/predictions_TIMESTAMP*

### Cardiac Diagnosis
1. Extract Features from the training dataset by running: generate_cardiac_features_train.py 
2. Extract Features from the testing dataset by running: generate_cardiac_features_test.py
3. Run the scripts stage_1_diagnosis.py and stage_2_diagnosis.py in sequence
4. The final cardiac disease prediction results on the test set are generated in the *prediction* folder


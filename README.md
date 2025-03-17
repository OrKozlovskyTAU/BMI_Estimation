# BMI Estimation Using Deep Learning

## Overview
This project presents a deep learning-based approach to estimate Body Mass Index (BMI) from a single 2D image. Our model is designed to be lightweight and efficient, making it suitable for deployment on mobile devices.

## Key Features
- **Deep Learning-based BMI Estimation**: Utilizes EfficientNet-B2 as the primary model architecture to extract deep feature representations from input images, with a series of 7 layers (FC + GELU) processing this deep features into a single scalar representing BMI.

<div align=center>
<img src="https://github.com/OrKozlovskyTAU/BMI_Estimation/blob/main/figures/architecture.png">
</div>

- **Data Augmentation & Image Segmentation**: Enhances dataset diversity and focuses the model on human body structure.
- **Hybrid Classification-Regression Approach**: To further refine our BMI estimation approach, we explored a hybrid classification-regression model. Initially, we trained a classifier to categorize images into one of five BMI-based classes and then we applied a regression model within each class to predict a precise BMI value.
- **Lightweight & Deployable**: The model is designed to run efficiently on constrained devices such as mobile phones.

## Dataset
The original dataset used in this project, *2D-image-to-BMI*, was scrapped from Reddit by [Jin et al.](https://ieeexplore.ieee.org/document/9699418) and contains:
- **4,189 images** (1,477 male and 2,712 female subjects)
- **Frontal view RGB images** with random backgrounds
- **Ground-truth BMI values** calculated from reported height and weight.

It can be downloaded from the [BaidudetDisk](https://pan.baidu.com/s/1HkFk3NCUtMSEDbTkkhULsA) with the code `FVL1`, or from the [Google Driver](https://drive.google.com/file/d/11P1NvO9cAM62TGgtwbPv9iUGjsx7b6IA/view?usp=sharing).

The dataset used in this project was derived from the original dataset after enhanced through the following steps:
- **Data Augmentation**: Horizontal flipping, Gaussian noise, rotation (up to 15 degrees), and random erasing.
- **Image Segmentation**: Uses Mask2Former to isolate human figures from backgrounds.

These enhancements expanded the dataset to **10,893 images**. 
The dataset used in this project can be found in this [Google Drive](https://drive.google.com/drive/folders/1ZNCCEEj7J_l8r4CetZC40JoAnkoWo_1-?usp=sharing).

<div align=center>
<img src="https://github.com/OrKozlovskyTAU/BMI_Estimation/blob/main/figures/data_bmi_histogram.png">
</div>

## Results
The following figure compares the performance of our base model with the previous approach by [Jin et al.](https://ieeexplore.ieee.org/document/9699418) and the hybrid classification-regression model. All results are evaluated using the mean absolute error (MAE) which measures absolute prediction errors and mean absolute percentage error (MAPE) that normalizes errors relative to ground truth.

<div align=center>
<img src="https://github.com/OrKozlovskyTAU/BMI_Estimation/blob/main/figures/performance_comparison.png">
</div>

Our model (Ours | Base, Red Bar) demonstrated an improvement over the previous approach by [Jin et al.](https://ieeexplore.ieee.org/document/9699418), reducing the Mean Absolute Error (MAE) from **3.96** to **3.66** and the Mean Absolute Percentage Error (MAPE) from **13.31%** to **11.78%**.
The hybrid approach (Ours | Hybrid, Yellow Bar), which introduced classification before regression, did not improve results; instead, MAE increased to 4.1 and MAPE slightly rose to 13.2%.  But, the hybrid approach using manual classification (Ours | Hybrid | Manual Classification, Green Bar), significantly reduced MAE to 1.31 and MAPE to 3.73%, demonstrating that accurate class assignments enhance regression precision.

## Installation

The code has been tested with PyTorch 2.2.1, CUDA 12.4, and Python 3.12.8. It may work with other versions.

### Step 1: Clone the Repository
To get started, clone the project's repository by running the following command:

```
git clone https://github.com/OrKozlovskyTAU/BMI_Estimation.git
cd BMI_Estimation
```

### Step 2: Set Up the Environment
To install the required dependencies, create and activate a new Conda environment:

```
conda create -n <env_name> python==3.12.8
conda activate <env_name>
pip install -r requirements.txt
```

Replace `<env_name>` with your preferred environment name.

### Step 3: Download Dataset and Pretrained Model
Download the dataset and pretrained model from [Google Drive](https://drive.google.com/drive/folders/1ZNCCEEj7J_l8r4CetZC40JoAnkoWo_1-?usp=sharing) and:

- Extract the dataset into the project directory. Ensure that the folder **Org_Aug_Seg_sorted** is placed at the project root.
- Move the pretrained `.pth` file into a directory named `./saved_pth/`. If the `saved_pth` folder does not exist, create it manually.

## Training & Evaluation

### Training

To train the model, run the following command:
```bash
python Train.py --model EfficientNet_Model --folder_path ./Org_Aug_Seg_sorted
```

#### Command Line Arguments
The `Train.py` script accepts several optional arguments to customize the training process:
- `--model` (default: `EfficientNet_Model`): Specifies the model architecture. Available options include:
  - `ResNet_Model`
  - `EfficientNet_Model`
  - `MobileNet_Model`
  - `VitTransformer_Model`
  - `CVT_Transformer_Model`
  - `DenseNet_Model`
  - `RegNet_Model`
- `--folder_path` (default: `./Org_Aug_Seg_sorted`): Path to the dataset folder containing training and test data.
- `--classification_activation`: Enables the hybrid approach for training. If not set, the model will follow the base solution.
- `--regression_after_classification`: Only applicable when `--classification_activation` is enabled. If set, applies regression within each class after classification; otherwise, only classification is performed.
- `--manual_classes`: Only applicable when `--classification_activation` is enabled. If set, manual classification is used instead of training a classifier.
- `--add_false`: Only applicable when `--classification_activation` is enabled. If set, incorporates false images from different classes to enhance generalization.

### Evaluation
To reproduce the results from the paper, first follow the installation instructions to obtain the pre-trained model.
This model was trained using the following hyperparameters:

- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 30
- **Loss Function**:
  - Mean Squared Error (MSE) for regression
  - CrossEntropyLoss for classification
- **activation function**: GELU (Gaussian Error Linear Unit)
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau

To evaluate the model, run:
```bash
python Test.py --folder_path ./Org_Aug_Seg_sorted --model_path ./saved_pth/model_Org_Aug_Seg_sorted.pth 
```
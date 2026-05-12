# PH AQI Classification ML

This repository contains the implementation for a machine learning project that classifies Air Quality Index (AQI) severity levels in Philippine urban environments using pollutant concentration data.

## Manuscript Title

**Classifying Air Quality Index Levels Based on Pollutant Concentrations in Philippine Urban Environments Using Support Vector Machine, LightGBM, CatBoost, and Multilayer Perceptron Neural Network**

## Project Overview

Air pollution remains an environmental and public health concern in Philippine urban areas. This project applies machine learning models to classify AQI severity levels using pollutant concentration features from Philippine city-level air quality records.

The study uses the **PH Philippine Cities Air Quality Index Data 2025** dataset from Kaggle. The target variable is `main.aqi`, which represents OpenWeather’s 1-to-5 AQI severity scale. Since the target is categorical, the task is formulated as a **supervised multiclass classification problem** rather than continuous AQI regression.

## Dataset

Dataset used:

**PH Philippine Cities Air Quality Index Data 2025**  
Kaggle Dataset: https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2025

The dataset is organized into monthly CSV files for 2025. These files are merged into a single master dataset during preprocessing.

### Dataset Columns Used

| Variable | Description |
|---|---|
| `datetime` | Timestamp of the air quality record. Used to derive temporal features such as month, day of week, and hour. |
| `main.aqi` | Target variable representing the AQI severity class. |
| `components.co` | Carbon monoxide concentration. |
| `components.no` | Nitrogen monoxide concentration. |
| `components.no2` | Nitrogen dioxide concentration. |
| `components.o3` | Ozone concentration. |
| `components.so2` | Sulfur dioxide concentration. |
| `components.pm2_5` | PM2.5 concentration. |
| `components.pm10` | PM10 concentration. |
| `components.nh3` | Ammonia concentration. |
| `city_name` | Philippine city associated with the air quality record. |

## Models Implemented

The following models were implemented and compared:

1. **Support Vector Machine**
   - Used as the baseline model.
   - Implemented as a margin-based classifier.

2. **LightGBM Classifier**
   - Gradient boosting model for structured tabular data.
   - Achieved the best overall performance in the study.

3. **CatBoost Classifier**
   - Gradient boosting model with categorical feature handling.
   - Used to compare against LightGBM and other models.

4. **Multilayer Perceptron Neural Network**
   - Neural network model using ReLU hidden layers and softmax output.
   - Trained using the Adam optimizer and sparse categorical cross-entropy loss.

## Methodology

The implementation follows this pipeline:

1. Download dataset using KaggleHub.
2. Locate and merge monthly `CombinedData` CSV files.
3. Inspect dataset structure and class distribution.
4. Remove invalid AQI labels and standardize column names.
5. Convert `datetime` and extract temporal features.
6. Define predictor variables and target variable.
7. Split dataset using stratified train-validation-test split.
8. Train SVM, LightGBM, CatBoost, and MLP-NN models.
9. Evaluate models using classification metrics.
10. Generate visualizations such as confusion matrices and feature importance plots.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- Normalized Confusion Matrix

Macro F1-score was emphasized because the dataset contains imbalanced AQI severity classes.

## Final Test Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| LightGBM Classifier | 0.9969 | 0.9599 | 0.9817 | 0.9704 | 0.9970 |
| CatBoost Classifier | 0.9926 | 0.9118 | 0.9591 | 0.9338 | 0.9927 |
| MLP Neural Network | 0.9493 | 0.8926 | 0.8013 | 0.8381 | 0.9489 |
| Support Vector Machine | 0.8461 | 0.6853 | 0.5687 | 0.6010 | 0.8431 |

LightGBM achieved the highest performance across all major metrics.

## Key Findings

- Gradient boosting models performed best for AQI severity classification.
- LightGBM achieved the highest Macro F1-score and overall classification performance.
- CatBoost also performed strongly but ranked slightly below LightGBM.
- MLP-NN performed acceptably but did not outperform the boosting models.
- SVM served as the baseline and had the lowest Macro F1-score.
- Feature importance results showed that pollutant concentration variables were more influential than temporal or city-based features.

## Repository Structure

```text
ph-aqi-classification-ml/
│
├── README.md
├── requirements.txt
├── CSELEC2C_Group_4_Implementation.ipynb
├── CSELEC2C_Group_4_Implementation.pdf
├── CSELEC2C_Group_4_Manuscript.pdf
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open CSELEC2C_Group_4_Implementation.ipynb.
2. Run all notebook cells in order.
3. The notebook will download the Kaggle dataset using KaggleHub, merge the monthly files, preprocess the data, train the models, evaluate performance, and generate result figures.

## Reproducibility Notes
- The dataset is downloaded directly from Kaggle using KaggleHub.
- A fixed random seed is used where applicable.
- A stratified 70/15/15 train-validation-test split is used.
- Results, metrics, and figures are saved under the outputs/ directory.

## License

This repository is intended for academic use as part of a machine learning final project.

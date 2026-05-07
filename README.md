# Irrigation Need Prediction - Machine Learning Ensemble

A comprehensive machine learning project for predicting irrigation needs using an ensemble of GPU-accelerated gradient boosting models (XGBoost, LightGBM, and CatBoost).

## 🎯 Project Overview

This project implements a sophisticated machine learning pipeline to predict irrigation requirements based on various environmental and soil factors. The system uses:

- **3 GPU-Accelerated Models**: XGBoost, LightGBM, and CatBoost
- **Advanced Feature Engineering**: Including ET₀ (Evapotranspiration) calculations, moisture deficit analysis, and soil water stress indicators
- **Composite Scoring**: Balances ROC-AUC and normalized PR-AUC for optimal performance on imbalanced data
- **Out-of-Fold Stacking**: Prevents data leakage in meta-model training
- **Weighted Ensemble**: Combines predictions using composite scores as weights

## � Model Performance

The ensemble achieves excellent performance on the holdout test set:

| Model | ROC-AUC | F1-Score | PR-AUC |
|-------|---------|----------|--------|
| XGBoost | 0.9346 | 0.7656 | 0.8711 |
| LightGBM | 0.9372 | 0.7809 | 0.8718 |
| CatBoost | 0.9176 | 0.7159 | 0.8350 |
| **Ensemble** | **0.9342** | **0.7669** | **0.8682** |

## 🚀 Features

### Data Processing
- **Comprehensive Feature Engineering**: 
  - Soil characteristics (pH, organic carbon, electrical conductivity)
  - Moisture metrics (soil moisture, rainfall, previous irrigation)
  - Climate variables (temperature, humidity, wind speed, sunlight)
  - Engineered features (moisture deficit, ET₀, water stress indicators)
  
- **Advanced Preprocessing**:
  - Target encoding for categorical variables
  - Ordinal encoding for ordered categories
  - Standard scaling for numerical features
  - Proper handling of categorical features for CatBoost

### Model Training
- **Hyperparameter Optimization**: Optuna-based tuning with 3-fold stratified cross-validation
- **GPU Acceleration**: All three models utilize NVIDIA CUDA for faster training
- **Class Imbalance Handling**: 
  - Sample weights for XGBoost
  - Class weights for LightGBM
  - Auto class weights for CatBoost

### Ensemble Methods
1. **Weighted Soft Voting**: Uses composite scores as weights
2. **Out-of-Fold Stacking**: Trains meta-models on OOF predictions to prevent leakage

## 📁 Project Structure

```
.
├── main.ipynb                          # Main Jupyter notebook with complete pipeline
├── best_hyperparameters.json           # Optimized hyperparameters for each model
├── best_scores.json                    # Composite scores for weighted ensemble
├── consensus_feature_importance.csv    # Feature importance rankings
├── consensus_feature_importance.json   # Detailed feature importance data
└── README.md                           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM

### Required Packages

```bash
pip install numpy pandas scikit-learn
pip install xgboost lightgbm catboost
pip install optuna matplotlib seaborn tqdm
pip install jupyter notebook
```

### GPU Setup

For GPU acceleration, ensure you have:
- NVIDIA GPU drivers installed
- CUDA Toolkit (version 11.0+)
- cuDNN library

The models will automatically detect and use GPU if available.

## � Usage

### Running the Complete Pipeline

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook main.ipynb
```

2. **Execute All Cells**: Run all cells in sequence to:
   - Load and preprocess data
   - Engineer features
   - Find optimal alpha for composite scoring
   - Optimize hyperparameters (or load existing ones)
   - Train final models
   - Evaluate on holdout set
   - Generate ensemble predictions

### Key Sections in the Notebook

1. **Data Loading & Preprocessing** (Cells 1-10)
   - Loads irrigation dataset
   - Handles missing values
   - Creates train/test split

2. **Feature Engineering** (Cells 11-25)
   - Implements `IrrigationFeatureEngineer` class
   - Calculates ET₀, moisture deficit, water stress
   - Creates interaction features

3. **Feature Selection** (Cells 26-35)
   - Defines feature lists
   - Builds preprocessing pipeline
   - Prepares data for modeling

4. **Optimal Alpha Calculation** (Cell 36)
   - Finds optimal balance between ROC-AUC and PR-AUC
   - Uses fast LightGBM proxy model

5. **Hyperparameter Optimization** (Cell 37)
   - Optuna-based tuning for all 3 models
   - 3-fold stratified CV with composite scoring
   - GPU-accelerated training
   - Saves results to JSON files

6. **Final Model Training** (Cell 38)
   - Trains models with best hyperparameters
   - Uses full training set
   - GPU acceleration enabled

7. **Model Evaluation** (Cell 39)
   - Evaluates each model on holdout set
   - Computes ROC-AUC, F1-Score, PR-AUC
   - Generates classification reports

8. **Ensemble Predictions** (Cells 40-41)
   - Weighted soft voting ensemble
   - Out-of-fold stacking with meta-models
   - Final performance comparison

## 🔧 Configuration

### Hyperparameter Tuning

To re-run hyperparameter optimization:

1. Delete existing JSON files:
```python
import os
os.remove('best_hyperparameters.json')
os.remove('best_scores.json')
```

2. Run the Optuna cell (Cell 37)

### Adjusting Number of Trials

In the Optuna cell, modify:
```python
models_to_optimize = [
    ("XGBoost",  objective_xgboost,  30),  # Change 30 to desired number
    ("LightGBM", objective_lightgbm, 30),
    ("CatBoost", objective_catboost, 30),
]
```

### GPU Configuration

The notebook automatically detects GPU. To force CPU:
```python
gpu_available = False
```

## 📊 Feature Importance

The project includes consensus feature importance across all models. Top features include:

1. **Moisture Deficit**: Gap between field capacity and current moisture
2. **Effective Rainfall**: Rainfall adjusted for runoff and deep percolation
3. **ET₀ (Evapotranspiration)**: Reference evapotranspiration rate
4. **Soil Moisture**: Current soil water content
5. **Temperature**: Ambient temperature affecting water needs

## 🧪 Model Details

### XGBoost
- **GPU**: CUDA-accelerated histogram-based algorithm
- **Imbalance**: Sample weights per fold
- **Early Stopping**: 20 rounds on validation set

### LightGBM
- **GPU**: Native GPU support via OpenCL
- **Imbalance**: Class weight dictionary
- **Early Stopping**: 20 rounds with callbacks

### CatBoost
- **GPU**: Task type set to GPU
- **Imbalance**: Auto class weights (Balanced)
- **Categorical Features**: Native handling without encoding

## 🎓 Methodology

### Composite Scoring

Instead of optimizing for ROC-AUC alone, we use a composite score:

```
Composite = α × ROC-AUC + (1-α) × PR-AUC-normalized
```

Where:
- α is optimized using a fast proxy model (typically 0.35-0.55)
- PR-AUC is normalized to remove baseline bias
- This balances overall discrimination with minority class precision

### Out-of-Fold Stacking

To prevent data leakage:
1. Generate OOF predictions for each base model using 5-fold CV
2. Stack OOF predictions as meta-features
3. Train meta-models (Logistic Regression, Random Forest, Gradient Boosting)
4. Evaluate on holdout set predictions

## � Results Interpretation

### ROC-AUC
- Measures overall discrimination ability
- Values > 0.93 indicate excellent performance
- Macro-averaged across all irrigation need classes

### F1-Score
- Harmonic mean of precision and recall
- Values > 0.76 show good balance
- Macro-averaged for multi-class problem

### PR-AUC
- Precision-Recall Area Under Curve
- More informative for imbalanced data
- Values > 0.87 indicate strong minority class performance

## 🐛 Troubleshooting

### GPU Not Detected
```python
# Check GPU availability
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

### Out of Memory
- Reduce batch size in model training
- Decrease number of Optuna trials
- Use fewer CV folds (change from 3 to 2)

### Slow Training
- Ensure GPU is being used (check cell outputs)
- Reduce number of estimators in hyperparameter search space
- Use fewer Optuna trials

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{irrigation_prediction_2024,
  title={Irrigation Need Prediction - ML Ensemble},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/irrigation-prediction}
}
```

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- XGBoost, LightGBM, and CatBoost teams for excellent gradient boosting implementations
- Optuna team for the hyperparameter optimization framework
- Scikit-learn for preprocessing and evaluation tools

---

**Note**: This project requires a dataset with irrigation-related features. Ensure your data includes soil characteristics, moisture levels, climate variables, and irrigation history for best results.

# Project Structure

This document describes the organization and purpose of files in this repository.

## Directory Tree

```
irrigation-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI/CD pipeline
│
├── main.ipynb                              # Main Jupyter notebook (complete pipeline)
├── best_hyperparameters.json               # Optimized hyperparameters for each model
├── best_scores.json                        # Composite scores for weighted ensemble
├── consensus_feature_importance.csv        # Feature importance rankings
├── consensus_feature_importance.json       # Detailed feature importance data
│
├── README.md                               # Project overview and documentation
├── PROJECT_STRUCTURE.md                    # This file
├── LICENSE                                 # MIT License
├── requirements.txt                        # Python dependencies
└── .gitignore                              # Git ignore rules
```

## File Descriptions

### Core Files

#### `main.ipynb`
**Purpose**: Complete machine learning pipeline from data loading to model evaluation.

**Sections**:
1. **Data Loading & Preprocessing** (Cells 1-10)
   - Load irrigation dataset
   - Handle missing values
   - Create train/test split

2. **Feature Engineering** (Cells 11-25)
   - `IrrigationFeatureEngineer` class
   - ET₀ calculations
   - Moisture deficit analysis
   - Water stress indicators

3. **Feature Selection** (Cells 26-35)
   - Define feature lists
   - Build preprocessing pipeline
   - Prepare data for modeling

4. **Optimal Alpha Calculation** (Cell 36)
   - Find optimal ROC-AUC/PR-AUC balance
   - Uses fast LightGBM proxy

5. **Hyperparameter Optimization** (Cell 37)
   - Optuna-based tuning
   - 3-fold stratified CV
   - GPU-accelerated training

6. **Final Model Training** (Cell 38)
   - Train with best hyperparameters
   - Full training set
   - GPU acceleration

7. **Model Evaluation** (Cell 39)
   - Holdout set evaluation
   - ROC-AUC, F1-Score, PR-AUC
   - Classification reports

8. **Ensemble Predictions** (Cells 40-41)
   - Weighted soft voting
   - OOF stacking
   - Performance comparison

**Key Classes**:
- `IrrigationFeatureEngineer`: Custom transformer for domain-specific features
- `detect_gpu()`: Automatic GPU detection
- `get_sample_weights()`: Balanced class weights
- `objective_*()`: Optuna objective functions

**Dependencies**: See `requirements.txt`

---

#### `best_hyperparameters.json`
**Purpose**: Stores optimized hyperparameters for each model.

**Structure**:
```json
{
  "XGBoost": {
    "xgb_n_estimators": 122,
    "xgb_max_depth": 9,
    "xgb_learning_rate": 0.1749,
    ...
  },
  "LightGBM": { ... },
  "CatBoost": { ... }
}
```

**Usage**: Loaded automatically if present; skips Optuna optimization.

---

#### `best_scores.json`
**Purpose**: Stores composite scores for weighted ensemble.

**Structure**:
```json
{
  "XGBoost": 0.9322,
  "LightGBM": 0.9348,
  "CatBoost": 0.9160
}
```

**Usage**: Used as weights in ensemble averaging.

---

#### `consensus_feature_importance.csv`
**Purpose**: Feature importance rankings across all models.

**Columns**:
- `feature`: Feature name
- `consensus_score`: Averaged importance across models
- `rank`: Overall ranking

**Top Features**:
1. Moisture_Deficit
2. Effective_Rainfall
3. ETo_approx
4. Soil_Moisture
5. Temperature_C

---

#### `consensus_feature_importance.json`
**Purpose**: Detailed feature importance data.

**Structure**:
```json
{
  "xgboost_top_15": { "feature": importance, ... },
  "lightgbm_top_15": { ... },
  "consensus_scores": { ... }
}
```

---

### Documentation Files

#### `README.md`
**Purpose**: Main project documentation.

**Contents**:
- Project overview
- Features and capabilities
- Performance metrics
- Installation instructions
- Usage examples
- Model details
- Troubleshooting guide
- Citation information

---

#### `PROJECT_STRUCTURE.md`
**Purpose**: This file - describes project organization.

**Contents**:
- Directory tree
- File descriptions
- Code structure
- Design patterns
- Extension guide

---

### Configuration Files

#### `requirements.txt`
**Purpose**: Python package dependencies.

**Categories**:
- Core: numpy, pandas, scikit-learn
- Models: xgboost, lightgbm, catboost
- Optimization: optuna
- Visualization: matplotlib, seaborn
- Utilities: tqdm, jupyter

**Installation**: 
```bash
pip install -r requirements.txt
```

---

#### `.gitignore`
**Purpose**: Specifies files to exclude from Git.

**Excluded**:
- Python cache (`__pycache__/`)
- Virtual environments (`.venv/`)
- Jupyter checkpoints
- IDE files (`.idea/`, `.vscode/`)
- Model outputs (`catboost_info/`)
- OS files (`.DS_Store`)

---

#### `LICENSE`
**Purpose**: MIT License for open-source distribution.

**Permissions**:
- ✓ Commercial use
- ✓ Modification
- ✓ Distribution
- ✓ Private use

**Conditions**:
- Include license and copyright notice

---

#### `.github/workflows/ci.yml`
**Purpose**: GitHub Actions CI/CD pipeline.

**Jobs**:
1. **Test**: Run on multiple Python versions
2. **Lint**: Check code style

**Triggers**:
- Push to main/develop
- Pull requests to main

---

## Code Organization

### Main Notebook Structure

```python
# 1. IMPORTS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# ... other imports

# 2. CONFIGURATION
RANDOM_SEED = 11
TEST_SIZE = 0.1

# 3. CUSTOM CLASSES
class IrrigationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for irrigation features"""
    
    def fit(self, X, y=None):
        # Calculate target means for encoding
        return self
    
    def transform(self, X):
        # Create engineered features
        return X_transformed

# 4. HELPER FUNCTIONS
def detect_gpu():
    """Detect available GPU"""
    pass

def get_sample_weights(y):
    """Calculate balanced class weights"""
    pass

# 5. DATA LOADING
df = pd.read_csv('irrigation_data.csv')

# 6. FEATURE ENGINEERING
engineer = IrrigationFeatureEngineer()
X_engineered = engineer.fit_transform(X_train, y_train)

# 7. PREPROCESSING
preprocessor = ColumnTransformer([...])
X_processed = preprocessor.fit_transform(X_engineered)

# 8. OPTIMAL ALPHA
OPTIMAL_ALPHA = find_optimal_alpha(X_train, y_train)

# 9. HYPERPARAMETER OPTIMIZATION
def objective_xgboost(trial):
    # Define search space
    # Train with CV
    # Return composite score
    pass

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgboost, n_trials=30)

# 10. FINAL TRAINING
final_models = {}
for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
    model = train_model(best_params[model_name])
    final_models[model_name] = model

# 11. EVALUATION
for model_name, model in final_models.items():
    evaluate_model(model, X_test, y_test)

# 12. ENSEMBLE
ensemble_pred = weighted_average(predictions, weights)
```

---

## Data Flow

```
Raw Data
    ↓
[Data Loading & Cleaning]
    ↓
[Feature Engineering]
    ↓
[Preprocessing Pipeline]
    ↓
[Train/Test Split]
    ↓
[Optimal Alpha Calculation]
    ↓
[Hyperparameter Optimization] → best_hyperparameters.json
    ↓                            best_scores.json
[Final Model Training]
    ↓
[Model Evaluation]
    ↓
[Ensemble Predictions]
    ↓
Final Results
```

---

## Key Design Patterns

### 1. Scikit-learn Pipeline
- Custom transformers inherit from `BaseEstimator` and `TransformerMixin`
- Ensures consistent preprocessing in train/test

### 2. Composite Scoring
- Balances ROC-AUC and PR-AUC
- Optimal for imbalanced data

### 3. Out-of-Fold Stacking
- Prevents data leakage
- Uses CV predictions for meta-features

### 4. GPU Abstraction
- Automatic GPU detection
- Graceful fallback to CPU

### 5. Configuration Management
- Hyperparameters stored in JSON
- Easy to version and share

---

## Extending the Project

### Adding New Models

1. Create objective function:
```python
def objective_newmodel(trial):
    # Define hyperparameters
    # Train with CV
    # Return composite score
    pass
```

2. Add to optimization list:
```python
models_to_optimize.append(
    ("NewModel", objective_newmodel, 30)
)
```

3. Add to final training:
```python
new_model = NewModel(**best_params["NewModel"])
new_model.fit(X_train, y_train)
final_models["NewModel"] = new_model
```

### Adding New Features

1. Modify `IrrigationFeatureEngineer`:
```python
def transform(self, X):
    X = X.copy()
    # Add new feature
    X['new_feature'] = calculate_new_feature(X)
    return X
```

2. Update feature lists:
```python
engineered_numeric.append('new_feature')
```

---

## Model Serialization

### Saving Models

```python
import joblib

# Save all models
joblib.dump(final_models, 'models.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(engineer, 'feature_engineer.pkl')
```

### Loading Models

```python
import joblib

# Load models
final_models = joblib.load('models.pkl')
preprocessor = joblib.load('preprocessor.pkl')
engineer = joblib.load('feature_engineer.pkl')
```

### Making Predictions

```python
# Load new data
new_data = pd.read_csv('new_irrigation_data.csv')

# Apply feature engineering
new_data_engineered = engineer.transform(new_data)

# Preprocess
new_data_processed = preprocessor.transform(new_data_engineered)

# Predict with each model
predictions = []
for model_name, model in final_models.items():
    if model_name == 'CatBoost':
        # CatBoost uses different data format
        pred = model.predict_proba(new_data_catboost.values)
    else:
        pred = model.predict_proba(new_data_processed)
    predictions.append(pred)

# Ensemble prediction
weights = [best_scores[name] for name in final_models.keys()]
ensemble_pred = np.average(predictions, axis=0, weights=weights)
final_pred = np.argmax(ensemble_pred, axis=1)
```

---

## Performance Optimization

### GPU Utilization

Monitor GPU usage during training:
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### Memory Management

For large datasets:
```python
# Use data chunking
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Parallel Processing

Utilize multiple cores:
```python
# In model parameters
n_jobs = -1  # Use all available cores
```

---

## Testing

### Unit Tests

Create `tests/test_features.py`:
```python
import unittest
from main import IrrigationFeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def test_moisture_deficit(self):
        # Test moisture deficit calculation
        pass
    
    def test_et0_calculation(self):
        # Test ET0 calculation
        pass
```

Run tests:
```bash
python -m unittest discover tests
```

---

## Deployment

### API Endpoint (Flask)

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models at startup
models = joblib.load('models.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process and predict
    prediction = make_prediction(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Container

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

Build and run:
```bash
docker build -t irrigation-prediction .
docker run -p 8888:8888 irrigation-prediction
```

---

## Version Control

### Branching Strategy
- `main`: Stable releases
- `develop`: Development branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes

### Commit Messages
```
feat: Add new feature
fix: Fix bug in preprocessing
docs: Update README
refactor: Improve code structure
test: Add unit tests
```

---

## File Sizes

| File | Approximate Size |
|------|-----------------|
| main.ipynb | 2-5 MB |
| README.md | 15 KB |
| PROJECT_STRUCTURE.md | 10 KB |
| requirements.txt | 1 KB |
| best_hyperparameters.json | 2 KB |
| best_scores.json | 200 B |
| consensus_feature_importance.csv | 2 KB |
| consensus_feature_importance.json | 5 KB |
| LICENSE | 1 KB |
| .gitignore | 500 B |
| ci.yml | 2 KB |

**Total Repository Size**: ~5-8 MB

---

## Quick Reference

### Installation
```bash
git clone https://github.com/yourusername/irrigation-prediction.git
cd irrigation-prediction
pip install -r requirements.txt
```

### Run Notebook
```bash
jupyter notebook main.ipynb
```

### Train Models
```python
# Delete existing hyperparameters to retrain
import os
os.remove('best_hyperparameters.json')
os.remove('best_scores.json')

# Run Optuna cell in notebook
```

### Make Predictions
```python
# Load models and predict (see Model Serialization section)
```

---

**Last Updated**: May 7, 2024  
**Version**: 1.0.0

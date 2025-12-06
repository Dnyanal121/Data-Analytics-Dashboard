# Data Visualization for Large Datasets

## Group Members
Dnyanal Deshmukh : 25-27-10
Rahul Prasad : 25-27-28
Moirangthem Famthoi : 25-14-08

---


**Project Overview**
This project is a comprehensive Data Science Dashboard built with Flask. It is designed to handle the end-to-end machine learning lifecycle—from raw data ingestion to final prediction—specifically optimized for large datasets using Dask library. The application provides an  GUI for performing statistical analysis, feature engineering, and model training without writing code.

---

## Project Modules & Workflow

The application follows a sequential workflow divided into five main sections:

### 1. Basic Data Analysis (EDA)
Provides an immediate statistical overview of the uploaded dataset.
* **Data Overview:** Calculates Count, Mean, Max, Min,standard deviation etc.
* **Data Cleaning:** Allows the user to interactively drop specific columns.

### 2. Feature Engineering
A comprehensive suite of tools to clean and prepare data for modeling:
* **i) Outlier Detection:**
    * Z-Score
    * Isolation Forest
    * Interquartile Range (IQR)
* **ii) Data Imbalance Handling:**
    * Random Oversampling
    * Random Undersampling
    * SMOTE (Synthetic Minority Over-sampling Technique)
* **iii) Transformation:**
    * *Scaling:* MinMax Scaler, Standard Scaler, Robust Scaler
    * *Encoding:* One-Hot Encoding, Label Encoding
    * *Log Transform:* For skewed distributions
* **iv) Feature Selection:**
    * Correlation Heatmap
    * ANOVA
    * Recursive Feature Elimination (RFE)
    * Random Forest Importance
* **v) Dimensionality Reduction:**
    * Principal Component Analysis (PCA)

### 3. Visualization and Analysis
Interactive plotting powered by Plotly and Seaborn for deep insights:
* **i) Univariate Analysis:** Histogram, Density Plot, Box Plot, Violin Plot, Count Plot, Pie Chart.
* **ii) Bivariate Analysis:** Scatter, Bubble, Line, Bar, Hexbin, Regression Plot, Joint Plot.
* **iii) Multivariate Analysis:** Correlation Heatmap, Parallel Coordinates, Pair Plot, 3D Scatter Plot.
* **iv) Time Series Analysis:** Time Series Line Plot, Seasonal Decomposition, Autocorrelation Function (ACF), Rolling Mean, Candlestick Plot.

### 4. Modeling
Train and evaluate various machine learning algorithms:
* **i) Regression:** Linear Regression, SVR,Random Forest Regressor.
* **ii) Classification:** Logistic Regression, SVC, KNN Classifier, Decision Tree Classifier.

### 5. Prediction
Deploys the trained model to generate predictions on new inputs based on the saved model artifacts.

---

## Dependencies
This project requires Python 3.8+ and the following libraries:(Recommended Python Version 3.10)

* **Core Framework:** `Flask`
* **Data Processing:** `numpy`, `pandas`, `dask[dataframe]`, `fsspec`, `pyarrow`, `openpyxl`
* **Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`
* **Statistics:** `scipy`, `statsmodels`

---

## Procedure to Run the Code

### Step 1: CExtract the Project
 extract it to your local directory.


### Step 2: Set up Environment
Open a terminal in the project folder. It is recommended to create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
For creating conda environment
```bash
conda create --name myenv python=3.10 -y
conda activate myenv
```

### Step 3: Install Dependencies
Run the following command to install all necessary packages
```bash
pip install -r requirements.txt
```

### Step 4 :Run the Application
Execute the main Flask application file using Python
```bash
python app.py
```



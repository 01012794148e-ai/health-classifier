# 🏥 Health Classification App (Healthy vs Sick)

Welcome to the **Health Classification App**, an end-to-end Deep Learning web application designed to classify physiological states into **Healthy** or **Sick** using clinical and multi-feature tabular data. 

This repository features a complete, interactive machine learning pipeline that parses MATLAB data files, processes the features through a pre-fitted scaler, and performs real-time classification using a trained Deep Learning model.

---

##  Project Overview

The core goal of this project is to provide an accessible and fast diagnostic support tool. The system accepts a MATLAB data file (`.mat`), processes the raw matrix, and outputs high-confidence predictions regarding the subject's health status.

### **How it Works:**
1. **File Upload:** The user uploads a `.mat` file containing clinical features.
2. **Dynamic Processing:** The application automatically extracts the valid data matrix and verifies if it meets the input requirement of exactly **52 features**. If the matrix is transposed (52 rows instead of columns), the app automatically handles the transposition.
3. **Scaling & Inference:** Features are normalized using a pre-saved `StandardScaler` (`scaler.pkl`), and then passed to a deep neural network model (`classification_model.keras`) trained using **TensorFlow/Keras**.
4. **Result Visualization:** The interface displays the final predicted status (**Healthy** or **Sick**) accompanied by a precise **Confidence Percentage**.

---

## 🛠️ Tech Stack & Libraries

The project is built using a modern Python-based machine learning stack:

* **User Interface:** [Streamlit](https://streamlit.io/) (for building a responsive, clean web UI)
* **Deep Learning Framework:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
* **Data Processing & Parsing:** [SciPy](https://scipy.org/) (specifically `scipy.io.loadmat` for `.mat` files), [NumPy](https://numpy.org/), and [Pandas](https://pandas.pydata.org/)
* **Preprocessing:** [Scikit-learn](https://scikit-learn.org/) (for feature standardization)
* **Model Serialization:** [Joblib](https://joblib.readthedocs.io/) (for exporting/loading the Scaler)

---

##  Repository Structure

The project directory is structured as follows:

```text
├── .streamlit/
│   └── config.toml             # Streamlit configuration settings
├── app.py                      # Main Streamlit web application source code
├── classification_model.keras  # Pre-trained Keras Deep Learning model
├── scaler.pkl                  # Fitted StandardScaler object (Joblib format)
├── requirements.txt            # System and library dependencies
├── runtime.txt                 # Specified Python environment (Python 3.10.11)
└── README.md                   # Project documentation (this file)

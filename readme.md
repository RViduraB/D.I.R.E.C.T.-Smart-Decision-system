# 🌾 D.I.R.E.C.T. Smart Decision System
### National Paddy & Rice Production Forecast System for Sri Lanka

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.2+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

An advanced decision-support framework powered by a **Custom Artificial Neural Network (ANN) built entirely from scratch (First Principles)**. This system monitors, analyzes, and forecasts district-wise paddy yields and evaluates macro-level national rice surpluses or deficits against Sri Lanka's annual demand threshold.

---

## 📌 1. Introduction & Problem Statement

### The Core Crisis
Sri Lanka facing an inability to accurately monitor, predict, and manage the national supply and demand of rice. This unpredictable nature directly threatens national food security and causes severe financial volatility for both local farmers and macro-economic planners.

### Current Bottlenecks
* **Inadequate Analytical Models:** Traditional statistical methods fail to map the complex, non-linear interactions of dynamic environmental factors like Rainfall, Soil pH, and Pest Damage simultaneously.
* **Information Delay:** The lack of an early-stage, data-driven yield tracking mechanism leads to sudden, unexpected rice deficits (requiring costly, emergency imports) or unmanaged surpluses (resulting in massive resource wastage).

---

## 🎯 2. Project Objectives

* **Granular Yield Forecasting:** Build a machine learning architecture to accurately predict district-wise paddy yields by processing localized soil chemistry, rainfall patterns, and specific seed varieties.
* **Strategic Decision Support:** Automate the transformation pipeline converting paddy yield into actual rice equivalents (via a dynamic $0.68$ conversion factor) and compare aggregates against the national annual demand threshold (**2.4 Million MT**) to instantly trigger `SURPLUS` or `DEFICIT` flags.
* **Modern Digital Deployment:** Deliver an interactive, user-centric **Streamlit Dashboard** allowing policy makers and agricultural officers to run real-time "What-If" simulations for future timelines (e.g., Year 2027).

---

## 🛠️ 3. Tech Stack & Engineering Highlights

* **Core Engine:** Pure Python 3 & NumPy (No heavy deep learning frameworks like TensorFlow or PyTorch).
* **UI/Dashboard:** Streamlit (For highly interactive charts, dynamic KPI cards, and custom data testing blocks).
* **Data Pipeline:** Pandas & Scikit-Learn (`LabelEncoder` for categorical strings, `StandardScaler` for mathematical normalization).
* **Model Serialization:** Joblib (To store weights, scaling states, and encoding transformers).

---

## 🧠 4. Proposed Methodology & AI Architecture
[Input Features: 7] -> [Dense Layer: 12 Neurons] -> [ReLU Activation] -> [Dense Layer: 1 Neuron] -> [Output Yield]

### 1. Mathematical Pipeline & Preprocessing
* **Categorical Encoding:** `LabelEncoder` translates textual variables (Districts and Seed Varieties) into indexable numerical sequences.
* **Feature Scaling:** `StandardScaler` standardizes features to zero-mean and unit-variance, ensuring optimal gradient flow and protecting the model from mathematical bias:
    $$z = \frac{x - \mu}{\sigma}$$

### 2. From-Scratch Neural Network Design
* **Dense Layer Infrastructure:** Implemented customized matrix dot-product operations with weights initialized via He-style scaling ($\sqrt{1/n}$) and zero-initialized bias units.
* **Activation Optimization:** Integrated a custom **ReLU (Rectified Linear Unit)** layer to filter negative structural noise while capturing highly complex non-linear crop features:
    $$f(x) = \max(0, x)$$
* **Backpropagation Logic:** Optimizes mathematical weights and biases through explicit manual matrix calculus using the **Chain Rule** against a **Mean Squared Error (MSE)** loss function over **500 Epochs**.


---
## 5. 🖼️ System Workflow & Neural Network Topology
![D.I.R.E.C.T. System Architecture & Topology](/DIRECT%20Architecture.jpeg)

*Note: The system contains a modular 7-feature input vector mapped into a manual 12-neuron hidden matrix with optimized gradient states.*


## 📂 6. Repository & Architecture Structure

```text
myproject/
│
├── .vscode/                 # Editor configuration settings
├── dashboard/
│   └── app.py               # Streamlit application UI and sidebar controls
│
├── engine/
│   ├── data/
│   │   └── my_data.csv      # Localized historical agronomic dataset
│   ├── models/              # Serialized pipeline weights & transformers (.pkl)
│   │   ├── le_district.pkl
│   │   ├── le_seed.pkl
│   │   ├── network_weights.pkl
│   │   └── scaler.pkl
│   ├── main.py              # Training pipeline, evaluation, and logging framework
│   └── predict.py           # Feedforward inference module for new matrices
│
└── requirements.txt         # Project dependencies blueprint

```


## 🚀 7. Installation & How to Run
Follow these simple steps to set up and run the D.I.R.E.C.T. System locally:

Prerequisites
Make sure you have Python 3.9 or higher installed on your system.

### Step 1: Clone the Repository
```
git clone [https://github.com/your-username/direct-smart-decision-system.git](https://github.com/your-username/direct-smart-decision-system.git)
cd direct-smart-decision-system

```
### Step 2: Install Dependencies
Install all required libraries using pip:
```
pip install -r requirements.txt
```
### Step 3: Run the Model Training & Pipeline Engine
Execute the core script to preprocess data, train the custom neural network from scratch, and verify local predictions via the terminal:
```
python engine/main.py
```
### Step 4: Launch the Streamlit Dashboard UI
Launch the interactive web UI to monitor analytics and simulate predictive scenarios for 2027:
```
streamlit run dashboard/app.py
```
## 📊 8. Expected Results & System Capabilities

Optimized Model Convergence: The proprietary neural network steadily minimizes its loss matrix over 500 training cycles to guarantee robust and narrow prediction bounds.

Granular Visual Analytics: Generates multi-layered charts showcasing total paddy and rice distributions across production districts alongside structured data frames.

High-Visibility Macro Decision Box: Features a color-coded notification system inside the UI (Green for Exportable Surplus, Red for Required Imports) to provide immediate macro-level insights for national supply chains.

## 9 👥 Team Members - Group 12 (OUSL - COU5300)

| Registration Number | Name | Student Email |
| :--- | :--- | :--- |
| **123586766** | P.D.V.S. Edirisinghe |  |
| **123585511** | P.K. S.R. Pallebage |  |
| **123572396** | R.V.P.T.S. Bandara | |
| **523593570** | P.D.W.P. Abewickrama | |
| **623572889** | R.M.C.N. Rathnayaka | |
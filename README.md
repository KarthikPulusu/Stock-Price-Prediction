# Stock Price Prediction using Deep Learning

This project demonstrates how to predict stock closing prices using a deep learning regression model built with PyTorch.  The workflow covers data preprocessing, feature engineering, model training with different activation functions, model evaluation and performance visualization.

---

## **Project Workflow**

### 1. Data Preparation
- **Load Data:** Reads stock price data from a CSV file.
- **Imputation:** Handles missing values for both categorical and numerical columns.
- **Feature Engineering:** Extracts date-based features (day, month, year, day of week, month end).
- **Encoding:** Converts categorical variables (`Stock Code`, `Sector`) into numerical format using one-hot encoding.

### 2. Data Standardization & Splitting
- **Scaling:** Standardizes features and target (`Close Price`) using `StandardScaler`.
- **Train-Test Split:** Splits the dataset into training and testing sets.

### 3. PyTorch Dataset Preparation
- Converts NumPy arrays to PyTorch tensors.
- Creates a `DataLoader` for efficient batch training.

### 4. Model Architecture
- Defines a `StockPredictor` neural network with configurable activation functions (ReLU, Leaky ReLU, Tanh, Sigmoid).
- The model consists of two hidden layers.

### 5. Training & Evaluation
- Trains the model using the Adam optimizer and MSE loss.
- Compares the effect of different activation functions (Leaky ReLU vs. Sigmoid) on training loss and gradient norms.
- Evaluates model predictions on the test set using regression metrics.

### 6. Visualization
- Plots actual vs. predicted stock prices.
- Visualizes loss curves and gradient norms for different activations.
![Predicted Vs Actual price Image](media/predicted Vs Actual.png)

### 7. Model Evaluation Metrics:
Mean Squared Error (MSE): 73.0312
Root Mean Squared Error (RMSE): 8.5458
R² Score: 0.9112

### 8. Results & Findings

- **Leaky ReLU** activation outperforms **Sigmoid** in both loss reduction and gradient stability.
- Leaky ReLU helps avoid vanishing gradients, leading to better and faster learning.
- The model can predict stock closing prices with an accuracy of 91% after proper preprocessing and training.

---

## **Key Files**

- `Stock_Price_Prediction.ipynb` — Main Jupyter notebook containing all code, analysis, and plots.
- `stock_price_prediction_dataset.csv` — Input dataset (not included here).

---

## **How to Run**

1. **Install Requirements:**
    - Python 3.x
    - `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `seaborn`

2. **Open the Notebook:**
    - Launch Jupyter Notebook or VS Code.
    - Open `Stock_Price_Prediction.ipynb`.

3. **Run All Cells:**
    - Execute each cell in order for end-to-end results.

---

## **Project Structure**

```
.
├── Stock_Price_Prediction.ipynb
├── stock_price_prediction_dataset.csv
└── README.md
```

---

## **Acknowledgements**

- Dataset and problem inspired by common stock price prediction tasks.
- Built using open-source libraries: PyTorch, scikit-learn, pandas, matplotlib.

---

## **Author**

- Pulusu Karthik

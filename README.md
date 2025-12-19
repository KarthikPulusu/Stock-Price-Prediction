# 📈 Stock Market Prediction using Machine Learning & Sentiment Analysis

## 📌 Overview
This project predicts stock closing prices by combining historical market data with financial news sentiment analysis. It enhances traditional time-series forecasting by integrating transformer-based NLP (FinBERT) and optional LLM-based sentiment analysis to capture market psychology and news impact.

The system is designed using a production-style ML pipeline with rolling-window validation to avoid data leakage and ensure realistic performance.

---

## 🎯 Objectives
- Predict future stock closing prices using historical data  
- Improve prediction accuracy using financial news sentiment  
- Compare rule-based, transformer-based, and GenAI sentiment approaches  
- Build a realistic, interview-ready ML pipeline  

---

## 🔧 Technologies Used
- **Programming:** Python  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Machine Learning:** Scikit-learn (Random Forest Regressor)  
- **NLP & Transformers:** FinBERT (HuggingFace Transformers)  
- **Optional GenAI:** LLM-based sentiment classification  
- **Evaluation:** MAE, RMSE, R²  

---

## 🗂 Dataset Description

### 1️⃣ Stock Market Data
- Historical stock prices (Open, High, Low, Close, Volume)
- Time-indexed daily data

### 2️⃣ News Headlines
- Financial and market-related news headlines
- Aligned by date with stock price data

---

## 🧹 Data Preprocessing
- Handled missing values and duplicates  
- Converted dates to time-series index  
- Normalized numeric features where required  
- Merged stock data with sentiment scores based on date  

---

## 📊 Feature Engineering

### Technical Indicators
The following indicators were computed to capture price trends and momentum:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- On-Balance Volume (OBV)

### Sentiment Features
Each news headline is converted into sentiment scores and aggregated daily.

---

## 🧠 Sentiment Analysis Approaches

### ✅ 1️⃣ FinBERT (Primary Approach)
- Transformer-based model trained on financial text  
- Outputs Positive / Neutral / Negative sentiment  
- More accurate than rule-based approaches like VADER  
- Industry-accepted for financial NLP tasks  

### ✅ 2️⃣ LLM-Based Sentiment (Optional / GenAI)
- Uses prompt-based classification with large language models  
- Demonstrates GenAI and prompt engineering skills  
- Used for comparison or ensemble analysis  

---

## 🤖 Machine Learning Model
- **Model Used:** Random Forest Regressor  
- Handles non-linear relationships effectively  
- Robust to noise in financial data  

### 🔄 Rolling Window Strategy
- Trained only on past data  
- Tested on future unseen data  
- Prevents data leakage and simulates real-world deployment  

---

## 📈 Model Evaluation
The model is evaluated using standard regression metrics:

- **MAE (Mean Absolute Error)** – Average prediction error  
- **RMSE (Root Mean Squared Error)** – Penalizes large errors  
- **R² Score** – Variance explained by the model  

### Sample Results
MAE ≈ 41
RMSE ≈ 62
R² ≈ 0.84

These results indicate strong predictive performance while remaining realistic for volatile financial markets.

---

## 📉 Visualization
- Actual vs Predicted stock prices  
- Time-series plots for trend comparison  
- Error distribution analysis  

---

## 🚀 Key Achievements
- Integrated transformer-based financial sentiment analysis  
- Achieved R² ≈ 0.84 without data leakage  
- Built a feature-rich ML pipeline combining structured and unstructured data  
- Demonstrated NLP, ML, and GenAI concepts in a single project  

---

## 📚 What I Learned
- Time-series forecasting with rolling validation  
- Financial NLP using transformer models  
- Importance of avoiding data leakage in ML  
- Combining structured market data with unstructured text  
- Building realistic, production-oriented ML workflows  

---

## 🔮 Future Enhancements
- Deploy model using FastAPI or Flask  
- Add MLOps components (monitoring, retraining)  
- Integrate real-time news APIs  
- Experiment with LSTM / Transformer-based forecasting models  
- Deploy on cloud platforms (AWS / GCP / Azure)  

---

## 🧑‍💻 Author
**Karthik Pulusu**  
B.Tech | Data Science & Machine Learning  
Hyderabad, India

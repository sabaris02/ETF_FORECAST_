# 📈 India ETF ML Dashboard

A production-grade machine learning dashboard for predicting **Indian Gold and Silver ETF** daily returns using regression and classification models.
access the website on online at https://etfforecast-ldv6ihy9sqsp3ugjaphvid.streamlit.app/
---
for local deployment and development installation
## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
etf_ml_dashboard/
├── app.py                  # Streamlit UI — main entry point
├── data_loader.py          # yfinance data download + context merging
├── feature_engineering.py  # All feature construction (no lookahead)
├── models.py               # Model definitions, training, evaluation
├── backtest.py             # Trading strategy backtesting
├── utils.py                # Plotly charts, metric cards, export helpers
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🎯 ETFs Covered

| Name | Ticker | Exchange |
|------|--------|----------|
| Nippon India Gold BeES | `GOLDBEES.NS` | NSE |
| Nippon India Silver BeES | `SILVERBEES.NS` | NSE |

---

## 🧠 ML Approach

### Data
- **5 years** of daily OHLCV data via `yfinance` (auto-adjusted close)
- Market context: NIFTY 50 (`^NSEI`), USD/INR (`INR=X`), India VIX (`^INDIAVIX`)

### Features (60+)
- Returns: daily, log, lagged (1–5 days)
- Rolling stats: mean & std at 5, 10, 20, 50 windows
- Momentum: price and return momentum
- Technical: RSI-14, MACD, Bollinger Bands, EMA-20/50, ATR-14
- Volume: change, ratio vs 10-day mean
- Context: NIFTY/USDINR/VIX returns and levels

### Targets
- **Regression**: next-day return (shifted by -1, strictly no lookahead)
- **Classification**: 1 if next-day return > 0, else 0

### Models

| Task | Models |
|------|--------|
| Regression | Linear Regression, Random Forest, Gradient Boosting |
| Classification | Logistic Regression, Random Forest, Gradient Boosting |

### Validation
- **TimeSeriesSplit** (3–10 folds, configurable) — respects temporal order
- StandardScaler applied only to linear models via sklearn Pipeline

---

## 📊 Dashboard Sections

1. **Market Overview** — live price stats + interactive candlestick chart with optional technical indicators
2. **Model Performance** — CV metrics (RMSE, MAE, R², Directional Accuracy for regression; Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix for classification)
3. **Feature Analysis** — bar chart + heatmap of feature importances
4. **Backtest Performance** — equity curves, Sharpe ratio, max drawdown, win rate vs buy-and-hold
5. **Export** — download OOF predictions as CSV

---

## 💰 Trading Strategy

```
if prediction > 0 (or class == 1):
    → Long (hold) — earn actual next-day return
else:
    → Cash — return = 0
```

**Performance metrics**: Cumulative return, Annualised return, Sharpe ratio (risk-free = 6.5%), Max drawdown, Win rate.

---

## ⚙️ Configuration (Sidebar)

| Setting | Options |
|---------|---------|
| ETF | Gold / Silver |
| Task | Regression / Classification |
| Algorithm | 3 models per task |
| CV Folds | 3–10 |
| Show Indicators | RSI, EMA, Bollinger Bands toggle |
| Walk-Forward | Rolling retraining backtest |
| Save Model | Persist to `trained_models/` via joblib |

---

## 📌 Assumptions

1. **Adjusted close prices** are used throughout (splits/dividends adjusted)
2. **Long-only strategy** — no shorting or leverage
3. **No transaction costs** or slippage modelled
4. **End-of-day execution** — signal generated after close, executed at next open (simplified to next close return)
5. Context data (NIFTY, VIX, USDINR) is forward-filled on non-overlapping trading days

---

## ⚠️ Limitations

1. **Financial data non-stationarity** — models may degrade over structural breaks
2. **Feature engineering** parameters are fixed (not optimised via hyperparameter search)
3. **No live trading integration** — purely for analysis and research
4. **OOF predictions** in backtesting may overestimate real performance (look-forward gap in TimeSeriesSplit)
5. **SILVERBEES** has shorter history than GOLDBEES — fewer data points available
6. **Not financial advice** — this tool is for educational and research purposes only

---

## 🛠 Tech Stack

- **Streamlit** — dashboard UI
- **yfinance** — market data
- **scikit-learn** — ML models, preprocessing, validation
- **ta** — technical indicators
- **Plotly** — interactive charts
- **joblib** — model persistence
- **pandas / numpy** — data processing

---

*Built for educational purposes. Always do your own research before making investment decisions.*

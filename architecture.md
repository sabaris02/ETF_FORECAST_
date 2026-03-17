# ETF Forecast Dashboard — Architecture

This document describes the high-level architecture, module breakdown, use cases, and data flow of the ETF ML Dashboard.

## System Architecture Diagram

```mermaid
flowchart TD
    %% External Data Source
    YF[Yahoo Finance API] --> |OHLCV & Context| DL[data_loader.py\n(Data Fetching & Alignment)]
    
    %% Core Modules
    DL --> |Raw DataFrame| FE[feature_engineering.py\n(Indicators & Targets)]
    FE --> |Features X, Target y| ML[models.py\n(Training & CV)]
    
    %% ML Pipeline Output
    ML --> |Trained Pipeline| ST[Trained Models on Disk]
    ML --> |OOF Predictions| BT[backtest.py\n(Trading Strategy)]
    
    %% Dashboard Application
    ML -.-> APP[app.py\n(Streamlit Dashboard)]
    BT -.-> APP
    FE -.-> APP
    
    %% Utils
    APP --- UT[utils.py\n(Plotly Charts & Formatters)]
    
    %% Users
    USER((User)) <--> |Config & Visuals| APP
```

## Data Flow

1. **Extraction**: `data_loader.py` fetches raw adjusted ETF prices and market context metrics (NIFTY50, USDINR, VIX) from Yahoo Finance.
2. **Transformation**: `feature_engineering.py` applies technical indicators (e.g., RSI, MACD, Bollinger Bands) using only past data. It computes the "next-day return" as the target variable (forward-shifted) to prevent look-ahead bias. NaN rows caused by rolling windows or shifting are dropped.
3. **Modeling**: `models.py` receives the cleaned `X` (features) and `y` (target). It trains either a regression or classification model using `TimeSeriesSplit` cross-validation to maintain chronological ordering.
4. **Evaluation**: 
    - The model pipeline outputs Out-Of-Fold (OOF) predictions.
    - `backtest.py` takes these OOF predictions, aligns them with the actual next-day returns, and calculates trading strategy metrics (cumulative return, Sharpe ratio, drawdown).
5. **Presentation**: `app.py` acts as the orchestrator and UI, using `utils.py` to render Plotly charts and format numeric outputs into a polished dashboard.

---

## Module Descriptions & Use Cases

### 1. `data_loader.py`
**Responsibility**: Extracts raw market data and aligns different time series.
**Use Cases**:
- Downloading 5 years of daily data for Gold or Silver ETFs.
- Fetching macroeconomic context data (NIFTY50, USDINR, VIX).
- Merging ETF data with context data using left-joins and forward-fills.

**Key Methods**:
- `download_etf_data(ticker, years)`: Fetches adjusted OHLCV data. Drops NaN closes and ffills gaps.
- `download_context_data(years)`: Fetches context indices/currencies.
- `get_full_dataset(etf_label)`: Master function that orchestrates downloading and joining all required data.

### 2. `feature_engineering.py`
**Responsibility**: Constructs stationary features for the ML models and defines the prediction target.
**Use Cases**:
- Adding momentum oscillators (RSI, MACD) and volatility bands (Bollinger).
- Creating lag features (AR terms) and rolling statistics.
- Defining the target variable (`regression_target` or `classification_target`).

**Key Methods**:
- `build_features(df)`: Master pipeline that sequentially applies all feature construction steps and drops NaN rows.
- `create_targets(df)`: Shifts the daily return forward (`shift(-1)`) so the model learns to predict *tomorrow's* return based on *today's* features.
- `get_X_y(df, feature_cols, task)`: Splits the final dataframe into the feature matrix and target vector.

### 3. `models.py`
**Responsibility**: Defines algorithms, handles cross-validation safely for time-series, and manages model persistence.
**Use Cases**:
- Training Linear/Logistic regression, Random Forests, or Gradient Boosting models.
- Evaluating model performance reliably using `TimeSeriesSplit`.
- Saving trained models to disk for later use.
- Running walk-forward validation.

**Key Methods**:
- `train_with_timeseries_split(X, y, model_name, task, n_splits)`: Iterates over time-series splits, aggregates OOF predictions, computes metrics per fold, and fits a final model on all data.
- `evaluate_regression/classification(...)`: Computes standard ML metrics (RMSE, R², F1, Accuracy, etc.).
- `walk_forward_train(...)`: Performs a walk-forward optimization simulation over a fixed rolling window.

### 4. `backtest.py`
**Responsibility**: Simulates a basic trading strategy based on model predictions.
**Use Cases**:
- Evaluating the economic value of the ML model vs a Buy-and-Hold baseline.
- Calculating risk-adjusted return metrics like the Sharpe Ratio and Max Drawdown.

**Key Methods**:
- `run_backtest(df, predictions, task, oof_indices)`: The core backtest engine. It aligns predictions with next-day returns properly to calculate daily strategy profit/loss, then computes cumulative wealth curves and risk metrics.
- `build_strategy_returns(...)`: Generates daily signal returns vector (Signal * Actual Return).

### 5. `utils.py`
**Responsibility**: UI helpers, standard formatters, and complex Plotly chart generation.
**Use Cases**:
- Standardizing the color palette and typography across the Streamlit app.
- Rendering interactive candlestick charts with technical overlays.
- Exporting raw prediction data to CSV.

**Key Methods**:
- `plot_price_chart(df, etf_label, show_indicators)`: Builds a multi-pane Plotly chart containing price, EMA, Bollinger Bands, and RSI.
- `plot_equity_curves(backtest_result)`: Plots the Cumulative Return area chart for the Strategy vs Baseline.
- `metric_card(...)`: Emits custom HTML/CSS for visually appealing dashboard metric tiles.

### 6. `app.py`
**Responsibility**: The user interface and main execution controller.
**Use Cases**:
- Allowing users to select parameters (Asset, Task type, Algorithm).
- Coordinating data fetching, training, and rendering the results progressively.
- Managing session state so navigation doesn't trigger unnecessary reloads.

**Key Configuration**:
- Built with Streamlit, using a dark theme with custom CSS for 'JetBrains Mono' font, status badges, and strict layout control.

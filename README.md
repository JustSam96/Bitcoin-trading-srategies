# Bitcoin Trading Strategy with Machine Learning

A Jupyter notebook that tries to predict a simple **buy / stay out** trading signal for Bitcoin using technical indicators and classic machine learning models.

> **Educational project only.** This is not financial advice, and the results here should not be used to trade real money. See [Known Limitations](#known-limitations-please-read) for important caveats about how the results should be interpreted.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [The idea in plain English](#the-idea-in-plain-english)
3. [Dataset](#dataset)
4. [Workflow](#workflow)
5. [Technical indicators explained](#technical-indicators-explained)
6. [Models compared](#models-compared)
7. [Results](#results)
8. [Known limitations (please read)](#known-limitations-please-read)
9. [Suggested improvements](#suggested-improvements)
10. [Getting started](#getting-started)
11. [Project structure](#project-structure)
12. [Glossary](#glossary)

---

## What this project does

The notebook (`bitcoin-strategy.ipynb`) walks through a complete, small machine learning pipeline on Bitcoin price data:

1. Loads about 2.8 million rows of one-minute Bitcoin price data.
2. Cleans the data (fills gaps, removes unusable rows).
3. Creates a **trading signal** (1 = buy / hold, 0 = stay out) from moving averages.
4. Builds **technical indicators** (EMA, ROC, Momentum, RSI, Stochastic Oscillator) to use as model inputs.
5. Compares **nine machine learning models** with cross-validation.
6. Tunes the best one (Random Forest) with grid search.
7. Evaluates it on held-out data and draws a simple backtest.

## The idea in plain English

Traders often watch two moving averages of the price: a **fast** one that reacts quickly and a **slow** one that shows the bigger trend.

- If the fast average is **above** the slow one, the price is trending up, so the signal is **1** (buy / hold).
- Otherwise the signal is **0** (stay out).

The question this notebook asks is: *can a machine learning model learn to predict this signal from other indicators like RSI, momentum and the stochastic oscillator?*

## Dataset

| Item | Details |
|---|---|
| Source | Kaggle: [`mlfinancebook/bitstamp-bicoin-minutes-data`](https://www.kaggle.com/datasets/mlfinancebook/bitstamp-bicoin-minutes-data) |
| File | `BitstampData.csv` |
| Exchange | Bitstamp (BTC/USD) |
| Frequency | One row per minute |
| Size | 2,841,377 rows, 8 columns |
| Period | Late 2011 to mid 2017 |

**Columns:**

| Column | Meaning |
|---|---|
| `Timestamp` | Time of the row as a Unix timestamp (seconds) |
| `Open`, `High`, `Low`, `Close` | Price at the start, highest, lowest and end of that minute |
| `Volume_(BTC)` | Amount of Bitcoin traded in that minute |
| `Volume_(Currency)` | Value traded in USD in that minute |
| `Weighted_Price` | Volume-weighted average price for that minute |

About 1.65 million of the 2.84 million rows contain real prices. The rest are gaps where no trades happened, which is why the notebook fills missing values.

The dataset is downloaded automatically with `kagglehub`, so you do not need to download it by hand.

## Workflow

```
Raw data  →  Clean  →  Create signal  →  Build indicators  →  Split data
                                                                   ↓
        Backtest  ←  Evaluate  ←  Tune best model  ←  Compare 9 models
```

### 1. Setup
Installs `scikeras` (the modern replacement for the old `keras.wrappers.scikit_learn`) and imports all libraries.

### 2. Load and inspect
Loads the CSV, checks its shape, previews the last rows with `tail()`, and shows summary statistics with `describe()`.

### 3. Clean
- Checks for missing values (there are many).
- **Forward fills** missing values (`ffill`): each gap is filled with the last known value above it. This makes sense for prices, since a price stays the same until a new trade happens.
- Drops the `Timestamp` column.

### 4. Create the target signal
```python
short_mavg = Close.rolling(10).mean()   # fast average (last 10 minutes)
long_mavg  = Close.rolling(60).mean()   # slow average (last 60 minutes)
signal     = 1 if short_mavg > long_mavg else 0
```

### 5. Build technical indicators
Adds EMA, ROC, Momentum, RSI, Stochastic Oscillator and Moving Averages over short, medium and long windows (10, 30 and 200 minutes). See the [next section](#technical-indicators-explained).

### 6. Tidy up
Removes rows with missing values (the first ~200 rows lack enough history for the 200-minute indicators).

### 7. Explore
- Line plot of the price
- Histograms of every feature
- Bar chart of how many `0` and `1` signals there are
- Correlation heatmap

### 8. Split the data
- Takes the **most recent 100,000 rows** to keep training fast.
- Splits them **80% training / 20% validation** (80,000 / 20,000 rows).

### 9. Compare models
Each model is scored with 10-fold cross-validation on the training set.

### 10. Tune, evaluate and backtest
Grid search on the Random Forest, then a final accuracy check, confusion matrix, feature importance chart and a cumulative returns plot.

## Technical indicators explained

| Indicator | The question it answers | Simple explanation |
|---|---|---|
| **EMA** (Exponential Moving Average) | Where is the price trending? | Like a normal average, but recent prices count more, so it reacts faster. |
| **MA** (Moving Average) | Where is the price trending? | The plain average of the last `n` closing prices. |
| **ROC** (Rate of Change) | How much has the price changed in percent? | A rise from 100 to 105 gives +5%. |
| **MOM** (Momentum) | How much has the price changed in dollars? | Today's price minus the price `n` minutes ago. |
| **RSI** (Relative Strength Index) | Has the price risen or fallen too fast? | A number from 0 to 100. Above about 70 is "overbought", below about 30 is "oversold". |
| **%K** (Stochastic) | Where is the price inside its recent range? | Near 100 means close to the recent high, near 0 means close to the recent low. |
| **%D** (Stochastic) | Same, but smoother. | A 3-period average of %K. |

Most indicators are computed with windows of **10, 30 and 200 minutes**, so the models can see both fast and slow behaviour.

## Models compared

| Short name | Model |
|---|---|
| LR | Logistic Regression |
| LDA | Linear Discriminant Analysis |
| KNN | K-Nearest Neighbours |
| CART | Decision Tree |
| NB | Gaussian Naive Bayes |
| NN | Neural Network (MLPClassifier) |
| AB | AdaBoost |
| GBM | Gradient Boosting |
| RF | Random Forest |

**How they are compared:** 10-fold cross-validation. The training data is split into 10 parts. The model trains on 9 parts and is tested on the 10th, repeating until every part has been the test set once. The average score (and its spread) shows how good and how stable each model is.

## Results

### Cross-validation accuracy (training data)

| Model | Mean accuracy | Std. dev. |
|---|---|---|
| **RF** | **0.9289** | 0.0026 |
| GBM | 0.9087 | 0.0024 |
| CART | 0.8941 | 0.0052 |
| LDA | 0.8909 | 0.0025 |
| AB | 0.8852 | 0.0040 |
| LR | 0.8621 | 0.0173 |
| NB | 0.7859 | 0.0063 |
| NN | 0.7853 | 0.0908 |
| KNN | 0.7331 | 0.0040 |

Random Forest came out on top, so it was selected for tuning.

### Grid search (Random Forest)

Settings tried: `n_estimators` in {20, 80}, `max_depth` in {5, 10}, `criterion` in {gini, entropy}.

**Best:** `criterion='gini'`, `max_depth=10`, `n_estimators=80`, with a cross-validation accuracy of **0.9022**.

### Final validation results (20,000 unseen rows)

**Accuracy: 90.6%**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 7,831 | 987 |
| **Actual 1** | 895 | 10,287 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (stay out) | 0.90 | 0.89 | 0.89 |
| 1 (buy / hold) | 0.91 | 0.92 | 0.92 |

### Most important features

The Random Forest relied mostly on **RSI30** (about 28% of importance), followed by **MOM30** (about 15%) and **ROC30** (about 14%), then the stochastic indicators. The raw price columns contributed very little.

## Known limitations (please read)

The headline accuracy of about 90% looks impressive, but several issues in the current notebook mean it **should not be read as evidence that the strategy would make money**. Being upfront about this is part of good practice, and fixing these is a great next step.

1. **The answer is partly leaked into the inputs.**
   The signal is defined as `short_mavg > long_mavg`. These two columns were meant to be removed before training, but the cell that drops them has a typo (`ataset=dataset.drop(...)` instead of `dataset=dataset.drop(...)`), so nothing was dropped. As a result `short_mavg`, `long_mavg`, `Open`, `High`, `Low` and `Volume_(Currency)` are still in the feature set (the feature importance chart shows them). Even though the Random Forest leaned on other columns, the model has access to the ingredients of the label, which inflates accuracy.

2. **Random train/test split on time series data.**
   `train_test_split` and shuffled `KFold` mix rows from different times. Neighbouring minutes are almost identical, so the model is tested on rows that sit right next to rows it trained on. Real trading only ever predicts the future from the past. `TimeSeriesSplit` or a simple "train on earlier, test on later" split is more honest.

3. **The backtest is not valid.**
   The validation rows are a random sample, not consecutive minutes, so `pct_change()` compares prices from unrelated moments. The cumulative "returns" of 700+ (and the near-perfect match between *Strategy* and *Actual* returns) are therefore an artefact, not real performance. A proper backtest needs consecutive rows, transaction costs, and slippage.

4. **The target is a trend label, not a future price move.**
   The signal describes what the average is *doing now*, not what the price will do *next*. Predicting it well does not mean predicting profitable trades. A more useful target would be something like "is the price higher N minutes from now?".

5. **Minor issues.**
   - `MA21`, `MA63` and `MA252` are named after 21/63/252 but actually use windows of 10, 30 and 200.
   - Accuracy alone can mislead if classes are imbalanced (here about 44% zeros and 56% ones in validation).
   - Logistic Regression and the neural network were run without scaling the features, which explains the convergence warnings and the unstable NN scores. Scaling is applied only in the grid search step, and then the final model is trained on unscaled data.
   - Only the latest 100,000 rows (about 70 days of minutes) are used, so the results describe a short slice of history.
   - Data ends in 2017, so it does not reflect how Bitcoin behaves today.

## Suggested improvements

- [ ] Fix the `ataset` typo and remove `short_mavg`, `long_mavg`, `Close`, and other price-level columns from the features.
- [ ] Use `TimeSeriesSplit` (or a chronological train/test split) instead of random splits.
- [ ] Define the target as a **future** move, for example the sign of the return over the next 10 or 60 minutes.
- [ ] Rebuild the backtest on consecutive rows with transaction costs, and compare against buy-and-hold.
- [ ] Add risk metrics: Sharpe ratio, maximum drawdown, win rate.
- [ ] Scale features in a `Pipeline` so scaling is applied correctly inside cross-validation.
- [ ] Try more recent data and daily or hourly bars.
- [ ] Tune more hyperparameters and try XGBoost or LightGBM.

## Getting started

### Requirements

- Python 3.9+
- Jupyter Notebook, JupyterLab, or Google Colab / Kaggle

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install numpy pandas matplotlib seaborn scikit-learn \
            keras tensorflow scikeras kagglehub jupyter
```

### Run it

```bash
jupyter notebook bitcoin-strategy.ipynb
```

Then run the cells from top to bottom.

### Important notes on running

- **Dataset path:** the notebook reads the data from `/kaggle/input/...`, which only exists on Kaggle. If you run it locally, change the `read_csv` line to use the `path` printed by `kagglehub.dataset_download(...)`, for example:
  ```python
  import os
  dataset = pd.read_csv(os.path.join(path, "BitstampData.csv"))
  ```
- **Kaggle account:** `kagglehub` may ask you to authenticate the first time.
- **Run time:** cross-validating nine models on 80,000 rows takes a few minutes, and the grid search a little longer. Lower `num_folds` or the row count for a quicker run.
- **Memory:** the full dataset uses roughly 170 MB before feature engineering, so a machine with 8 GB of RAM is comfortable.
- The Keras imports are included but no deep learning model is trained in the current version.

## Project structure

```
.
├── bitcoin-strategy.ipynb   # Main notebook (all code and outputs)
└── README.md                # This file
```

## Glossary

| Term | Meaning |
|---|---|
| **Feature** | An input column the model learns from (e.g. RSI30). |
| **Target / label** | The thing the model tries to predict (here, `signal`). |
| **Cross-validation** | Testing a model several times on different slices of data to get a reliable score. |
| **Overfitting** | A model that memorises the training data and performs worse on new data. |
| **Data leakage** | When information about the answer sneaks into the inputs, making results look better than they really are. |
| **Backtest** | Simulating how a strategy would have performed on historical data. |
| **Confusion matrix** | A table of correct and incorrect predictions for each class. |
| **Precision** | Of the times the model said "buy", how often it was right. |
| **Recall** | Of all the real "buy" cases, how many the model caught. |
| **Look-ahead bias** | Accidentally using future information to make a past decision. |

## Disclaimer

This repository is for learning purposes. Past performance does not predict future results, and cryptocurrency trading carries a high risk of loss. Nothing here is financial advice.

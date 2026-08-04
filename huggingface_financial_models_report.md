# HuggingFace Pre-Trained Models for Price Direction Prediction

## Executive Summary

Searched HuggingFace Hub across 20+ query categories. Found **~30 relevant models** across 4 categories. Below are the best options organized by sklearn compatibility.

---

## 1. TIME SERIES FORECASTING MODELS

### ⭐ mohan170802/stock-price-predictor-xgboost — **BEST FOR SKLEARN**
- **URL:** https://huggingface.co/mohan170802/stock-price-predictor-xgboost
- **Format:** XGBoost .pkl (pickle) and .json (native Booster)
- **Sklearn-compatible:** ✅ YES — load with `pickle.load()` or `xgboost.Booster()`
- **Purpose:** Predicts 5-day-ahead stock price direction (UP/DOWN) and log-return
- **Tickers:** SPY, QQQ, AAPL, MSFT, TSLA, NVDA, AMD, META, JPM, XOM
- **Accuracy:**
  - **Directional Accuracy: 56.83% ± 3.53%** (beats 50% random chance)
  - SPY: 61.4%, META: 60.1%, MSFT: 58.9%, AMD: 58.5%
- **Features:** 66 engineered features (lag features, returns, SMA/EMA, RSI, MACD, Bollinger Bands, ATR, OBV, candlestick patterns, calendar features)
- **Training:** 15 years daily OHLCV (2011-2026), walk-forward validation
- **Usage with sklearn:**
  ```python
  import pickle
  from huggingface_hub import hf_hub_download

  # Download model
  model_path = hf_hub_download(
      repo_id="mohan170802/stock-price-predictor-xgboost",
      filename="AAPL_clf_5d.pkl"
  )
  with open(model_path, 'rb') as f:
      model = pickle.load(f)

  # model is xgboost.Booster — use model.predict(X)
  # Features must match the 66-feature engineering pipeline
  ```

### EsferSami/Apple_Stock_Price_Prediction
- **URL:** https://huggingface.co/EsferSami/Apple_Stock_Price_Prediction
- **Format:** ARIMA .pkl, Box-Cox transformer .pkl, LSTM .h5, scaler .joblib
- **Sklearn-compatible:** ✅ Partially — ARIMA model is pickle-based
- **Purpose:** 7-day AAPL price forecast using ARIMA or LSTM
- **Accuracy:** Not reported in model card
- **Usage with sklearn:**
  ```python
  import pickle
  from huggingface_hub import hf_hub_download

  arima_path = hf_hub_download(
      repo_id="EsferSami/Apple_Stock_Price_Prediction",
      filename="Apple-Stock-Price-Forecasting-ARIMA-Model/apple_stock_arima.pkl"
  )
  with open(arima_path, 'rb') as f:
      arima_model = pickle.load(f)
  # Use arima_model for forecasting
  ```

### shagatoo/ARIMA_Stock
- **URL:** https://huggingface.co/shagatoo/ARIMA_Stock
- **Format:** ARIMA model .pkl
- **Sklearn-compatible:** ✅ YES — pickle-based
- **Purpose:** Time-series forecasting on Yahoo Finance stock data
- **Dataset:** bwzheng2010/yahoo-finance-data
- **Usage with sklearn:**
  ```python
  import pickle
  from huggingface_hub import hf_hub_download

  model_path = hf_hub_download(
      repo_id="shagatoo/ARIMA_Stock",
      filename="arima_model.pkl"
  )
  with open(model_path, 'rb') as f:
      arima_model = pickle.load(f)
  ```

### huggingface/time-series-transformer-tourism-monthly
- **URL:** https://huggingface.co/huggingface/time-series-transformer-tourism-monthly
- **Format:** PyTorch + safetensors (transformer model)
- **Sklearn-compatible:** ❌ Requires PyTorch
- **Downloads:** 1,949 | **Likes:** 31
- **Note:** Official HuggingFace time-series transformer. Good reference but needs PyTorch.

---

## 2. FINANCIAL SENTIMENT MODELS

### ⭐ ProsusAI/finbert — **GOLD STANDARD**
- **URL:** https://huggingface.co/ProsusAI/finbert
- **Downloads:** 5,645,185 | **Likes:** 1,211
- **Format:** PyTorch `.bin` + **TensorFlow `.h5`** + Flax
- **Sklearn-compatible:** ❌ Requires transformers library (but has TF backend)
- **Purpose:** Financial sentiment analysis (positive, negative, neutral)
- **Accuracy:** ~93% on Financial PhraseBank (from paper)
- **Paper:** https://arxiv.org/abs/1908.10063
- **Workaround for sklearn-only:** Use ONNX export from HuggingFace or use the TF model with tf.keras, then extract embeddings for sklearn
  ```python
  # With transformers (not pure sklearn but no PyTorch)
  from transformers import pipeline

  nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
  result = nlp("Stocks rallied and the British pound gained.")
  # → [{'label': 'positive', 'score': 0.9998}]
  ```

### ⭐ yiyanghkust/finbert-tone
- **URL:** https://huggingface.co/yiyanghkust/finbert-tone
- **Downloads:** 795,976 | **Likes:** 220
- **Format:** PyTorch `.bin` + **TensorFlow `.h5`**
- **Sklearn-compatible:** ❌ Requires transformers (has TF backend)
- **Purpose:** Financial tone analysis (positive, negative, neutral)
- **Trained on:** 4.9B tokens from 10-K, 10-Q, earnings calls, analyst reports
- **Accuracy:** Superior performance on financial tone (per paper)
- **Paper:** Huang et al. (2022) Contemporary Accounting Research
- **Usage:**
  ```python
  from transformers import BertTokenizer, BertForSequenceClassification, pipeline

  finbert = BertForSequenceClassification.from_pretrained(
      'yiyanghkust/finbert-tone', num_labels=3
  )
  tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
  nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

  results = nlp([
      "there is a shortage of capital",
      "growth is strong and we have plenty of liquidity",
      "there are doubts about our finances",
      "profits are flat"
  ])
  # LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
  ```

### ahmedrachid/FinancialBERT-Sentiment-Analysis
- **URL:** https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis
- **Downloads:** 329,106 | **Likes:** 98
- **Format:** PyTorch `.bin`
- **Sklearn-compatible:** ❌ Requires transformers
- **Purpose:** Financial sentiment (positive, negative, neutral)
- **Accuracy:**
  - **Macro F1: 0.98**, Weighted F1: 0.98
  - Negative: P=0.96, R=0.97, F1=0.97
  - Neutral: P=0.98, R=0.99, F1=0.98
  - Positive: P=0.98, R=0.97, F1=0.97
- **Training:** Financial PhraseBank, 4840 sentences

### mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- **URL:** https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- **Downloads:** 206,663 | **Likes:** 462
- **Format:** PyTorch `.bin` + safetensors
- **Sklearn-compatible:** ❌ Requires transformers
- **Purpose:** Financial news sentiment (positive, negative, neutral)
- **Accuracy:** **98.23% accuracy** on Financial PhraseBank (sentences_allagree)
- **Note:** Distilled RoBERTa — 2x faster than full RoBERTa-base

### Bencode92/tradepulse-finbert-sentiment
- **URL:** https://huggingface.co/Bencode92/tradepulse-finbert-sentiment
- **Downloads:** 11,443 | **Likes:** 0
- **Format:** safetensors
- **Sklearn-compatible:** ❌ Requires transformers
- **Purpose:** Trading-oriented financial sentiment
- **Accuracy:** **100%** on custom training set (1797 samples, 2 epochs)
  - ⚠️ Likely overfitted to small dataset
- **Labels:** negative, neutral, positive

### soleimanian/financial-roberta-large-sentiment
- **URL:** https://huggingface.co/soleimanian/financial-roberta-large-sentiment
- **Downloads:** 3,212 | **Likes:** 57
- **Purpose:** Sentiment on financial statements, ESG reports, earnings calls

---

## 3. TRADING-RELATED MODELS

### JonusNattapong/Reinforcement-Learning-for-Gold-Trading-Model
- **URL:** https://huggingface.co/JonusNattapong/Reinforcement-Learning-for-Gold-Trading-Model
- **Downloads:** 422 | **Likes:** 10
- **Format:** Stable-Baselines3 (PPO agent)
- **Sklearn-compatible:** ❌ Requires stable-baselines3
- **Purpose:** Gold price (XAUUSD) trading via reinforcement learning
- **Dataset:** ZombitX64/xauusd-gold-price-historical-data-2004-2025

### dannymarcos/aegis-trading-models
- **URL:** https://huggingface.co/dannymarcos/aegis-trading-models
- **Downloads:** 45 | **Likes:** 0
- **Format:** Keras
- **Sklearn-compatible:** ❌ Keras only

### Badumetsibb/conscious-trading-agent-models
- **URL:** https://huggingface.co/Badumetsibb/conscious-trading-agent-models
- **Downloads:** 1 | **Likes:** 0
- **Format:** Keras `.keras` + **joblib** scalers
- **Sklearn-compatible:** ⚠️ Scalers are joblib (sklearn-compatible), but main model is Keras
- **Files:** calibrated_scaler.joblib, multi_horizon_scaler.joblib

### AdityaaXD/Multi-Agent_Reinforcement_Learning_Trading_System_Models
- **URL:** https://huggingface.co/AdityaaXD/Multi-Agent_Reinforcement_Learning_Trading_System_Models
- **Format:** Stable-Baselines3 (DQN, PPO, A2C)
- **Sklearn-compatible:** ❌ Requires stable-baselines3

---

## 4. FEATURE EXTRACTION MODELS FOR FINANCIAL DATA

### joyjitroy/Stock_Market_News_Sentiment_Analysis — **SKLEARN NATIVE**
- **URL:** https://huggingface.co/joyjitroy/Stock_Market_News_Sentiment_Analysis
- **Library:** `sklearn` (tagged)
- **Sklearn-compatible:** ✅ YES — native sklearn model
- **Format:** Jupyter notebook with full sklearn pipeline
- **Purpose:** Embedding-based sentiment classification using gradient boosting
- **Features:** TF-IDF embeddings + gradient boosting for stock news sentiment
- **Tags:** finance, sentiment-analysis, embeddings, gradient-boosting, classical-ml, nlp
- **Paper:** https://ssrn.com/abstract=5784922
- **Usage:**
  ```python
  # Clone/notebook-based workflow with sklearn GradientBoosting
  from huggingface_hub import hf_hub_download

  notebook_path = hf_hub_download(
      repo_id="joyjitroy/Stock_Market_News_Sentiment_Analysis",
      filename="NLP_Stock_Sentiment_Analysis_Full_Code.ipynb"
  )
  # Run notebook cells to train/load the model
  ```

### yiyanghkust/finbert-pretrain (Feature Extraction)
- **URL:** https://huggingface.co/yiyanghkust/finbert-pretrain
- **Downloads:** 270,009 | **Likes:** 40
- **Purpose:** Pre-trained financial BERT — use for feature extraction
- **Task:** Fill-mask (pre-training objective)
- **Usage for embeddings:**
  ```python
  from transformers import BertTokenizer, BertModel
  import torch
  import numpy as np

  tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
  model = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')

  def get_financial_embedding(text):
      inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
      with torch.no_grad():
          outputs = model(**inputs)
      # Use [CLS] token embedding as feature vector
      return outputs.last_hidden_state[:, 0, :].numpy()

  # Then feed embeddings to sklearn classifiers
  ```

### nickmuchi/finbert-tone-finetuned-finance-topic-classification
- **URL:** https://huggingface.co/nickmuchi/finbert-tone-finetuned-finance-topic-classification
- **Downloads:** 89,302 | **Likes:** 69
- **Purpose:** Financial topic classification (use embeddings as features for downstream sklearn models)
- **Note:** Access-restricted, requires HF login

---

## 5. MODELS COMPATIBLE WITH SKLEARN (NO PYTORCH)

| Model | Format | Direct sklearn? | Purpose |
|-------|--------|-----------------|---------|
| **mohan170802/stock-price-predictor-xgboost** | .pkl (XGBoost) | ✅ YES | Price direction prediction (56.8% accuracy) |
| **EsferSami/Apple_Stock_Price_Prediction** | .pkl (ARIMA) | ✅ YES | AAPL price forecasting |
| **shagatoo/ARIMA_Stock** | .pkl (ARIMA) | ✅ YES | Stock price forecasting |
| **joyjitroy/Stock_Market_News_Sentiment_Analysis** | sklearn notebook | ✅ YES | News sentiment with gradient boosting |
| **ProsusAI/finbert** | .h5 (TensorFlow) | ⚠️ TF backend | Financial sentiment (use ONNX for sklearn) |
| **yiyanghkust/finbert-tone** | .h5 (TensorFlow) | ⚠️ TF backend | Financial tone analysis |

---

## 6. RECOMMENDED SKLEARN-ONLY PIPELINE

For **price direction prediction** without PyTorch:

### Step 1: Get Sentiment Scores (alternative approaches)
```python
# Option A: Use ProsusAI/finbert via ONNX Runtime (pip install onnxruntime)
# Export from HuggingFace or find ONNX-converted version
# Option B: Use a simple TF-IDF + sklearn classifier (joyjitroy approach)
# Option C: Skip sentiment, use only price features

# For pure sklearn sentiment (no deep learning):
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
# Train on Financial PhraseBank dataset
```

### Step 2: Use XGBoost for Price Direction
```python
import pickle
from huggingface_hub import hf_hub_download
import numpy as np

# Load the classification model
model_path = hf_hub_download(
    repo_id="mohan170802/stock-price-predictor-xgboost",
    filename="SPY_clf_5d.pkl"  # Choose your ticker
)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Replicate the 66-feature engineering pipeline from stock_predictor.py
# (included in the repo as a reference)
# Then predict direction:
# prediction = model.predict(xgb.DMatrix(features))
```

---

## KEY FINDINGS

1. **Best native sklearn model:** `mohan170802/stock-price-predictor-xgboost` — XGBoost models predicting 5-day price direction with **56.8% directional accuracy** across 10 major tickers, with .pkl files for direct sklearn usage.

2. **Best sentiment model (needs transformers):** `ProsusAI/finbert` — 5.6M downloads, industry standard for financial sentiment with positive/negative/neutral classification.

3. **Best accuracy for sentiment:** `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` — **98.23% accuracy** on Financial PhraseBank.

4. **Pure sklearn option:** `joyjitroy/Stock_Market_News_Sentiment_Analysis` — Gradient boosting on embeddings for stock news sentiment.

5. **No single pre-trained model combines sentiment + price features end-to-end** — the typical workflow is to use sentiment models for feature extraction, then feed those features into sklearn classifiers for price direction prediction.

6. **Price direction prediction accuracy ceiling:** The best XGBoost models achieve ~57% directional accuracy — modest but statistically significant above the 50% random baseline, consistent with efficient market hypothesis constraints.

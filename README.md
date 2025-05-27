# Fake News Classification – Detection and Analysis

## 🎯 Overview

A comprehensive machine learning pipeline for detecting fake news articles using traditional ML techniques (TF-IDF + classical classifiers) and advanced ensemble methods. This repository provides both a research framework and a practical implementation guide for fake news detection.

## 🚨 Critical Dataset Finding

During our analysis, we discovered that the 'subject' field in the dataset is **perfectly correlated** with the label:
- **Fake news sources**: Government News, Middle-east, News, US_News, left-news, politics
- **Real news sources**: politicsNews, worldnews

This represents significant data leakage that explains the extremely high model performance (95-100% accuracy).

## 📊 Key Results

### Model Performance Comparison

| Model | Features Used | Validation Type | Accuracy | Precision | Recall | F1 Score | Key Finding |
|-------|--------------|-----------------|----------|-----------|---------|----------|-------------|
| MultinomialNB | TF-IDF only | Random Split | 95.51% | 95.41% | 96.04% | 95.72% | Baseline content learning |
| Logistic Regression | TF-IDF only | Random Split | 98.78% | 99.08% | 98.57% | 98.83% | Strong vocabulary patterns |
| LinearSVC | TF-IDF only | Random Split | 99.60% | 99.60% | 99.64% | 99.62% | Near-perfect on random split |
| XGBoost (Baseline) | TF-IDF only | Random Split | 99.76% | 99.87% | 99.66% | 99.77% | Captures subtle patterns |
| Voting Ensemble | TF-IDF only | Random Split | 99.62% | 99.62% | 99.66% | 99.64% | No improvement over best model |
| **Enhanced XGBoost** | TF-IDF + Statistical | Random Split | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **Pure style classification** |
| Robust XGBoost | TF-IDF + Statistical | Cross-Source | 99.46% | 99.92% | 98.62% | 99.26% | Slight generalization drop |
| **Content-Only XGBoost** | **TF-IDF only** | **Cross-Source** | **99.67%** | **99.82%** | **99.29%** | **99.55%** | **Vocabulary still decisive** |
| Noisy Stats XGBoost | TF-IDF + Noisy Stats | Cross-Source | 99.50% | 99.92% | 98.74% | 99.32% | Marginal robustness gain |

### 🔍 Investigation Results

Our Enhanced XGBoost model achieved **100% accuracy** through:

1. **Statistical Feature Dominance** (Enhanced XGBoost only):
   - Character count: 22.98% importance
   - Word count: 24.49% importance  
   - Lexical diversity: 42.42% importance
   - Capital ratio: 10.11% importance
   - **All TF-IDF features combined: 0.00% importance**
   
   Note: This is specific to the Enhanced XGBoost model. Baseline models using only TF-IDF features show normal text-based feature importance.

2. **Writing Style Differences**: The model learned to detect publication sources rather than fake vs. real content

3. **Cross-Source Validation**: Even with content-only features, the model achieves 99.55% F1 score, indicating systematic vocabulary differences between fake and real sources

## 🏗️ Project Structure

```
news-classification/
├── data/                     # Raw CSVs (True.csv, Fake.csv)
├── src/                      # Python source code (pipeline modules)
│   ├── data/                 # Data loading & cleaning
│   ├── features/             # Text preprocessing & vectorization
│   ├── models/               # Model training & evaluation
│   └── analysis/             # EDA & feature importance
├── workdir/                  # Auto-generated artifacts
│   ├── processed_data*.parquet
│   ├── eda/                  # EDA plots & summary
│   ├── features/             # TF-IDF matrices + vectorizer
│   ├── models/               # Fitted .joblib model files
│   └── reports/              # Evaluation plots & metrics
├── notebooks/
│   ├── fake_news_classification_final.ipynb  # Complete analysis with robustness
│   └── fake_news_pipeline_overview.ipynb      # Original pipeline walkthrough
└── docs/                     # Sphinx documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS users: Install LLVM OpenMP runtime (`brew install libomp`)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/news-classification.git
cd news-classification

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Command-Line Pipeline

```bash
# 1. Data cleaning
python src/data/make_dataset.py \
  --true_path data/True.csv \
  --fake_path data/Fake.csv \
  --output_path workdir/processed_data.parquet

# 2. Text preprocessing
python src/features/preprocess_text.py \
  --processed_path workdir/processed_data.parquet \
  --output_path workdir/processed_data_clean.parquet

# 3. TF-IDF vectorization + train/test split
python src/features/vectorize_split.py \
  --clean_path workdir/processed_data_clean.parquet \
  --output_dir workdir/features

# 4. Train & evaluate models
python src/models/train_evaluate.py \
  --feature_dir workdir/features \
  --model_dir workdir/models \
  --report_dir workdir/reports

# 5. Feature importance analysis (optional)
python src/analysis/feature_importance.py \
  --vectorizer_path workdir/features/vectorizer.joblib \
  --model_dir workdir/models \
  --output_dir workdir/reports/importance
```

#### Option 2: Jupyter Notebook (Recommended for Analysis)

```bash
jupyter notebook notebooks/fake_news_classification_final.ipynb
```

This notebook includes:
- Complete baseline implementation
- Enhanced models achieving 100% accuracy
- **Investigation section** explaining why 100% accuracy was achieved
- **Robustness analysis** with cross-source validation
- **Recommendations** for real-world deployment

## 📈 Key Features

### 1. Data Pipeline
- Automated data loading and cleaning
- Text preprocessing (lowercasing, punctuation removal, stop-word removal, lemmatization)
- TF-IDF vectorization with configurable parameters
- Stratified train/test splitting

### 2. Model Suite
- **Traditional ML**: Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Ensemble Methods**: XGBoost, Voting Classifier
- **Enhanced Models**: Statistical feature engineering
- **Robust Models**: Cross-source validation

### 3. Analysis Tools
- Exploratory Data Analysis (EDA)
- Feature importance visualization
- Learning curves
- Confusion matrices
- Statistical feature analysis

### 4. Investigation & Robustness
- Data leakage detection
- Writing style analysis
- Cross-source validation
- Content-only classification
- Noise injection techniques

## 🛡️ Robustness Recommendations

### For Production Deployment

1. **Use Cross-Source Validation**: Hold out entire news sources for testing
2. **Remove Source-Specific Features**: Filter out publication-identifying terms
3. **Focus on Content**: Use only TF-IDF features, not statistical features
4. **Expected Performance**: 70-85% F1 for truly generalizable models
5. **DO NOT deploy the 100% accuracy model** - it won't generalize to new sources

### For Research

1. **Dataset Limitations**: This dataset is useful for studying source characteristics, not genuine fake news detection
2. **Alternative Datasets**: Consider FakeNewsNet, LIAR, or fact-checking datasets
3. **Feature Engineering**: Focus on claim verification rather than style
4. **Evaluation**: Always use cross-source or temporal validation

## 📊 Generated Artifacts

The pipeline creates several analysis files:

- `workdir/eda/` – Class distribution, text-length histograms
- `workdir/reports/metrics.json` – Model performance metrics
- `workdir/reports/model_f1_comparison.png` – F1 score comparison
- `workdir/reports/*confusion.png` – Confusion matrices
- `workdir/reports/importance/` – Feature importance visualizations
- `workdir/reports/*_learning_curve.png` – Learning curves

## 📚 Documentation

### Build HTML Documentation
```bash
sphinx-build -b html docs docs/_build/html
```

### View Pre-built Documentation
```bash
open docs/_build/html/index.html        # macOS
xdg-open docs/_build/html/index.html    # Linux
start docs\_build\html\index.html       # Windows
```

## 🔬 Technical Details

### Feature Extraction
- **TF-IDF Features**: 50,000 most frequent terms
- **Statistical Features**: 
  - Character count
  - Word count
  - Average word length
  - Punctuation count
  - Capital letter ratio
  - Lexical diversity
  - Average sentence length

### Model Hyperparameters
- **XGBoost Enhanced**: 
  - n_estimators=200
  - max_depth=7
  - learning_rate=0.1
  - colsample_bytree=0.8

### Cross-Source Split
- **Train Sources**: politicsNews, politics, News, Government News
- **Test Sources**: worldnews, left-news, Middle-east, US_News

## ⚠️ Important Findings

1. **Perfect Accuracy = Red Flag**: The 100% accuracy indicates severe overfitting to source-specific patterns
2. **Style vs. Content**: Models learn writing style, not fake news indicators
3. **Vocabulary Segregation**: Even content-only models achieve 99%+ accuracy due to systematic vocabulary differences
4. **Limited Applicability**: Models trained on this data will fail on sophisticated fake news

## 🎯 Conclusion

This project demonstrates the critical importance of:
- Understanding your data (identifying leakage)
- Proper validation strategies (cross-source, not just random)
- Being skeptical of "too good to be true" results
- Building models that generalize, not memorize

While the technical implementation achieves impressive metrics, the investigation reveals that genuine fake news detection requires more sophisticated approaches and better datasets where fake and real news sources cover the same topics with similar writing styles.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle
- Framework inspired by best practices in ML research and production deployment

---

**Note**: This repository serves as both an educational resource for understanding ML pipelines and a cautionary tale about the importance of proper validation in machine learning.
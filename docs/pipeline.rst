Pipeline walkthrough
=====================

The project is structured as a **linear sequence of reproducible scripts**; each step reads
inputs from *workdir* and writes outputs back to it.

.. mermaid::
   :caption: High-level data flow

   graph TD;
     A[Raw CSVs True.csv + Fake.csv] --> B[make_dataset.py];
     B --> C[processed_data.parquet];
     C --> D[preprocess_text.py];
     D --> E[processed_data_clean.parquet];
     E --> F[vectorize_split.py];
     F -.->|saves| G[features/ X_*.npz + y_*.npy];
     G --> H[train_evaluate.py];
     H --> I[models/*.joblib];
     H --> J[reports/plots+metrics];
     I --> K[feature_importance.py];
     H --> K;
     K --> L[importance/*.png + .csv];

Detailed steps
--------------

1. **Data loading & cleaning** – `src/data/make_dataset.py`
   * Adds binary label, removes nulls, merges *title* + *text* into *full_text*
   * Output: `workdir/processed_data.parquet`

2. **Exploratory analysis** – `src/analysis/eda.py`
   * Generates class distribution, token length histograms, subject counts.

3. **Text preprocessing** – `src/features/preprocess_text.py`
   * Lower-case, punctuation & digit removal, stop-word filtering (NLTK), WordNet lemmatisation.
   * Output: `..._clean.parquet` with `clean_text` column.

4. **Vectorisation & split** – `src/features/vectorize_split.py`
   * TF-IDF (max 50 k vocab, 1–2-grams), stratified `train_test_split` (80/20).
   * Saves: `X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy` and `vectorizer.joblib`.

5. **Model training & evaluation** – `src/models/train_evaluate.py`
   * Trains MultinomialNB, LogisticRegression, LinearSVC, and XGBoost.
   * Computes accuracy, precision, recall, F1, ROC-AUC; saves confusion matrices & ROC curves.

6. **Feature importance** – `src/analysis/feature_importance.py`
   * Linear models: signed coefficients; MultinomialNB: log-prob diff; XGBoost: gain-based importance.
   * Saves CSV + bar plots for top words.

All scripts accept command-line arguments so you can tweak paths and hyper-parameters. 
Quick-start
===========

.. warning::
   **Critical Finding**: The dataset contains significant data leakage. The 'subject' field is perfectly correlated with labels, leading to unrealistically high accuracy (95-100%). See the Investigation Results section for details.

Install dependencies
--------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # fish/zsh/bash
   pip install -r requirements.txt

(macOS only) XGBoost depends on the LLVM OpenMP runtime. Install it if you haven't already:

.. code-block:: bash

   brew install libomp

Run the pipeline
----------------

The commands below assume you have the raw CSVs in ``data/``.

.. code-block:: bash

   # Data cleaning & parquet
   python src/data/make_dataset.py --true_path data/True.csv --fake_path data/Fake.csv \
       --output_path workdir/processed_data.parquet

   # Text preprocessing (stop-words, stemming/lemmatisation)
   python src/features/preprocess_text.py \
       --processed_path workdir/processed_data.parquet \
       --output_path workdir/processed_data_clean.parquet

   # TF-IDF + train/test split
   python src/features/vectorize_split.py --clean_path workdir/processed_data_clean.parquet \
       --output_dir workdir/features  --max_features 50000  --ngram_min 1 --ngram_max 2

   # Train & evaluate models
   python src/models/train_evaluate.py --feature_dir workdir/features --model_dir workdir/models \
       --report_dir workdir/reports

   # Feature-importance
   python src/analysis/feature_importance.py --vectorizer_path workdir/features/vectorizer.joblib \
       --model_dir workdir/models --output_dir workdir/reports/importance

Interactive notebooks
---------------------

Two notebooks are available for different purposes:

1. **Original Pipeline Overview** (basic analysis):

   .. code-block:: bash

      jupyter notebook notebooks/fake_news_pipeline_overview.ipynb

2. **Complete Analysis with Investigation** (recommended):

   .. code-block:: bash

      jupyter notebook notebooks/fake_news_classification_final.ipynb

   This notebook includes:
   
   * Baseline models achieving 95-99% accuracy
   * Enhanced models with **100% accuracy**
   * Investigation revealing data leakage issues
   * Cross-source validation showing 99.5%+ accuracy even with content-only features
   * Robustness recommendations for real-world deployment

Key findings
------------

* **Data Leakage**: Subject field perfectly correlates with labels
* **Style vs. Content**: Models learn writing style, not fake news indicators  
* **Limited Applicability**: Models will fail on sophisticated fake news
* **Expected Real-World Performance**: 70-85% F1 for truly generalizable models 
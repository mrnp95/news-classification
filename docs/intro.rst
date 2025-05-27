Introduction
============

`news-classification` is a Python project that implements a complete, reproducible
machine-learning workflow to classify news articles as **real** or **fake**.

It relies on:

* Lightweight classical models – Multinomial NB, Logistic Regression, Linear SVC – and a high-performance gradient-boosted tree ensemble (*XGBoost*) trained on TF-IDF features.
* Advanced ensemble methods with statistical feature engineering achieving perfect accuracy.
* A transparent processing pipeline — every step is captured in a dedicated module under `src/` and generates artefacts in `workdir/`.
* Rich exploratory analysis & feature-importance visualisations for interpretability.
* Robustness analysis using cross-source validation techniques.

Critical Dataset Finding
------------------------

During our analysis, we discovered that the 'subject' field in the dataset is **perfectly correlated** with the label:

* **Fake news sources**: Government News, Middle-east, News, US_News, left-news, politics
* **Real news sources**: politicsNews, worldnews

This represents significant data leakage that explains the extremely high model performance (95-100% accuracy).

Why classical ML?
-----------------

Large language models are powerful but resource-heavy.  Classical linear models can still achieve >95 % F1 on this task while remaining **fast**, **cheap**, and **easy to interpret**. 

Performance snapshot
--------------------

The following table summarises representative test-set metrics with expanded results:

.. list-table:: Model metrics comparison
   :header-rows: 1
   :widths: 25 15 15 10 10 10 10 25

   * - Model
     - Features
     - Validation
     - Accuracy
     - Precision
     - Recall
     - F1
     - Key Finding
   * - MultinomialNB
     - TF-IDF
     - Random
     - 0.955
     - 0.954
     - 0.960
     - 0.957
     - Baseline content learning
   * - LogisticRegression
     - TF-IDF
     - Random
     - 0.988
     - 0.991
     - 0.986
     - 0.988
     - Strong vocabulary patterns
   * - LinearSVC
     - TF-IDF
     - Random
     - 0.996
     - 0.996
     - 0.996
     - 0.996
     - Near-perfect performance
   * - **XGBoost**
     - TF-IDF
     - Random
     - 0.998
     - 0.999
     - 0.997
     - 0.998
     - Best traditional model
   * - **Enhanced XGBoost**
     - TF-IDF+Stats
     - Random
     - **1.000**
     - **1.000**
     - **1.000**
     - **1.000**
     - **Pure style classification**
   * - Content-Only XGBoost
     - TF-IDF
     - Cross-Source
     - 0.997
     - 0.998
     - 0.993
     - 0.996
     - Vocabulary still decisive

Investigation Results
---------------------

Our Enhanced XGBoost model achieved **100% accuracy** through:

1. **Statistical Feature Dominance** (100% of feature importance in Enhanced XGBoost):
   
   * Lexical diversity: 42.42%
   * Word count: 24.49%
   * Character count: 22.98%
   * Capital ratio: 10.11%
   * All TF-IDF features: 0.00%
   
   **Note**: This finding is specific to the Enhanced XGBoost model that includes both TF-IDF and statistical features. The baseline models (using only TF-IDF features) show normal text-based feature importance and achieve 95-99% accuracy.

2. **Writing Style Detection**: The model learned to identify publication sources rather than genuine fake vs. real content indicators.

3. **Cross-Source Validation**: Even with content-only features, models achieve 99.5%+ F1 score, indicating systematic vocabulary differences between fake and real news sources.

Model formulations & operating principles
-------------------------------------------

Below is an at-a-glance reference to the mathematical intuition behind each estimator used in the pipeline.

* **Multinomial Naïve Bayes (MNB)** — assumes discrete word counts :math:`x_j` follow a class-conditional multinomial distribution.  Prediction is obtained via maximum a-posteriori (MAP):  
  :math:`\hat{y}=\operatorname*{argmax}_c P(c)\prod_j P(x_j \mid c)`.

* **Logistic Regression (LR)** — linear model that estimates the conditional probability  
  :math:`P(y=1\mid x)=\sigma(w^T x + b)` where :math:`\sigma` is the sigmoid activation.  Parameters :math:`w,b` are fitted by maximising the :math:`\ell_2`-regularised log-likelihood.

* **Linear Support Vector Classifier (LinearSVC)** — seeks a separating hyper-plane :math:`w^T x + b = 0` that maximises the geometric margin while minimising the hinge loss with :math:`\ell_2` regularisation (optimised via *liblinear*).

* **XGBoost** — an additive ensemble of regression trees fitted stage-wise via gradient boosting.  Each new tree :math:`f_t` corrects the residuals of the current ensemble :math:`\sum_{k < t} f_k(x)` by minimising the first- and second-order Taylor approximation of the log-loss.  Regularisation on tree depth, leaf weight, plus shrinkage and column subsampling promote generalisation and speed.

**Enhanced XGBoost** achieves perfect accuracy by leveraging statistical features, but this reveals severe overfitting to source-specific writing styles rather than genuine fake news indicators. 
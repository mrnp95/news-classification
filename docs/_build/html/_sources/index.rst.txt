.. news-classification documentation master file

news-classification documentation
==================================

Welcome to **news-classification**'s documentation!
This site describes the end-to-end pipeline for fake vs. real news detection — from data loading to model evaluation and feature interpretation.

.. warning::
   **Critical Finding**: Our analysis revealed perfect correlation between the 'subject' field and labels, indicating severe data leakage. Models achieve 95-100% accuracy by learning source-specific patterns rather than genuine fake news indicators. See the Investigation Results section for details.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   intro
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   pipeline
   learning_curves
   investigation
   api

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous

   license

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
# NCAA March Machine Learning Competition - Post-Competition Analysis

This repository contains the code and analysis for my attempt at the NCAA March Machine Learning Kaggle competition. While I was unable to submit my results before the deadline, I'm sharing my findings and methodologies for educational purposes and potential future improvements.

## Project Overview

This project focused on predicting the outcomes of the NCAA Men's and Women's Basketball Tournaments using machine learning. The core of the project was extensive feature engineering, leveraging regular season averages and relative performance metrics, considering factors like region, conference, and opponent strength.

## Key Findings

### Men's Tournament

* The best performing model for the men's tournament was an **XGBoost classifier enhanced with Graph Neural Network (GNN) embeddings**.
    * Model Parameters: `n_estimators=300`, `gamma=0.1`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.9`, `max_depth=3`.
    * GNN embeddings were generated using a network with `k=4`, 256 channels, and 31 embeddings.
    * Final Brier Score: **0.19361297357560175**
* While the GNN-enhanced XGBoost model provided a marginal improvement over baseline XGBoost and SVM models, its computational cost, particularly the GNN embedding generation, is a consideration.
* The small improvement in Brier score, however, could be the difference between winning and losing in a competition setting.

### Women's Tournament

* The best performing model for the women's tournament was a **KMeans-enhanced Support Vector Machine (SVM) classifier**.
    * Model Parameters: `C=1`, `gamma=0.01`, `kernel='rbf'`.
    * The KMeans cluster feature, treated as a continuous variable, significantly improved the SVM's performance.
    * Final Brier Score: **0.17297381393779618**
* Graph Neural Network (GNN) features did not improve the model's performance on the women's dataset.
* The results highlight the effectiveness of SVMs with carefully engineered features and optimized parameters for predicting women's tournament outcomes.

### Feature Engineering

* Extensive feature engineering, including relative performance metrics and regional/conference considerations, proved crucial for model performance.
* The feature engineering notebook explores the impact of engineered features on XGB Survival and XGB Classifier models, indicating the importance of these features.
* Future work will focus on adapting feature engineering techniques for datasets with limited data beyond DayNum 121, particularly for ranking features.

## Project Structure

* `data/`: Contains raw and processed data files (CSV).
* `notebooks/`: Jupyter notebooks for data exploration, feature engineering, and modeling.
    * **Feature_Engineering.ipynb:** Contains all the feature engineering and EDA.
    * **Modeling_NCAA_Men.ipynb:** Contains the modeling of the mens data.
    * **Modeling_NCAA_Women.ipynb:** Contains the modeling of the womens data.
    * **Final_Pipeline_And_Predictions.ipynb:** Contains all the functions for preprocessing and creating the features, creates the training data, uses the models to predict the outcomes. At the bottom you can insert team names and opponent names to see the predicted probabilities that one of the teams will win and then check google to evaluate accuracy.
* `scripts/`: Python scripts for reusable functions and model training.
* `models/`: Saved models and related artifacts.
* `reports/`: Reports and visualizations.
* `requirements.txt`: List of Python dependencies.
* `README.md`: This file.
* `.gitignore`: Specifies files to ignore by Git.

## Future Work

* Explore more efficient GNN embedding generation techniques.
* Investigate feature reduction methods to mitigate computational overhead.
* Experiment with alternative encoding strategies for KMeans cluster features.
* Further analyze feature interactions to improve model performance.
* Adapt feature engineering for datasets with data limitations.
* Thorough cross-validation for model evaluation.

## Dependencies

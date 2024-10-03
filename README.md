This project tackles a multiclass classification problem with a heavily imbalanced dataset, focusing on identifying various categories from an Intel dataset using advanced machine learning techniques. The goal was to handle the imbalance and achieve high performance using F1-score as the key metric.
📊 Dataset Overview

    Samples: 34,553 rows
    Features: 7,672 features (after preprocessing, reduced to 7,539 features)
    Challenge: Handling missing values, redundant features, and highly imbalanced classes

Key Issues Addressed:

    Handling Imbalanced Data: Utilized techniques like SMOTE (Synthetic Minority Oversampling Technique) for generating synthetic samples in minority classes.
    Dimensionality Reduction: Reduced the feature space using PCA and Kernel PCA to tackle the curse of dimensionality.
    Model Selection: Employed XGBoost, Random Forest, and CatBoost for optimal model performance.

⚙️ Preprocessing and Feature Engineering 🧑‍💻

    Imputation of Missing Values:
        Used K-Nearest Neighbor (KNN) imputation and median imputation to fill in missing data.
        Handled outliers using 3-sigma rule and IQR filtering.

    Feature Scaling:
        Features were scaled using RobustScaler, MinMaxScaler, and StandardScaler depending on the nature of the model.

    Dimensionality Reduction:
        Reduced the feature space using Principal Component Analysis (PCA) and Kernel PCA (kPCA) to retain the most valuable components while speeding up model training.
        Selected 1,500 components for PCA and 875 components for kPCA, which balanced dimensionality with performance.

🛠️ Modeling and Hyperparameter Tuning ⚡

We trained several models and performed hyperparameter tuning using Stratified kFold Cross-Validation to ensure robustness and fair evaluation. Below are the models and their performance:

    XGBoost:
        Best performing model with an F1-score of 0.978 🏅
        Key hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=500

    Random Forest:
        F1-score of 0.855, hyperparameters optimized via Grid Search.
        Key hyperparameters: max_depth=10, n_estimators=1000

    CatBoost:
        Another strong contender with an F1-score of 0.912

🔍 Model Evaluation and Results 📈

The models were evaluated based on the F1-score due to the imbalanced nature of the dataset. Here are the top-performing models:
Model	Average F1-Score
XGBoost	0.978
CatBoost	0.912
Random Forest	0.855

    XGBoost outperformed all other models, achieving the best results in both cross-validation and on the test set.
    Extensive grid search across different parameters helped optimize the results.

🏆 Conclusion and Future Work ✨

    XGBoost proved to be the best model for handling the imbalanced multiclass problem, achieving an F1-score of 0.978.
    SMOTE significantly helped balance the minority classes, allowing the models to better distinguish between different categories.
    Future work includes exploring deep learning models and further hyperparameter tuning to push the F1-score even higher.

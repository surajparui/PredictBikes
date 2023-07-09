### Report: Predict Bike Sharing Demand with AutoGluon Solution
#Author: Suraj M Parui

This repository contains the solution for the "Predict Bike Sharing Demand" project using AutoGluon. The goal of this project is to develop a machine learning model that accurately predicts the demand for bike sharing based on various features.

##Initial Training
During the initial stage of training, the predictions were carefully examined to identify any negative values. Fortunately, all the predictions were positive, eliminating the need to assign values to 0 before submission. Additionally, there was no need to convert the count column from float32 to int64 as the regression output was evaluated based on the root mean square logarithmic error (RMSLE) metric.

The top-ranked model that performed the best was the Weighted_Ensemble_L3 model. This model is a stack of three layers of previously trained models, designed to maximize the validation accuracy.

##Exploratory Data Analysis and Feature Creation
During the exploratory data analysis (EDA), several observations were made:

Certain features, such as "holiday" and "working day," were binary in nature.
Features like "temp," "atemp," "humidity," and "windspeed" exhibited a near-normal distribution.
Categorical features, such as "season" and "weather," were identified.
The data followed a monthly pattern across the years 2011 and 2012.
The "hour" feature was included in the dataset to capture the demand patterns throughout the day.
The inclusion of the "hour" feature significantly improved the performance of the model. It provided valuable information and insights to the trained models, allowing them to better understand the demand patterns at different times of the day.
![hour_feature_hist.png](img/hour_feature_hist.png)

## Hyper parameter tuning
After trying different hyperparameters, the model's performance improved. Although the training root mean square error (RMSE) increased slightly, the test error (RMSLE) decreased significantly. This suggests that the model may have a slightly higher bias but better variance, resulting in improved generalization on the test data.

By focusing on tree-based models with boosting ensemble techniques, such as Gradient Boosting, CATboost, and XGBoost, the model achieved better generalization and reduced the test error. This highlights the importance of selecting appropriate models and optimizing their hyperparameters for better performance.
### How much better did your model perform after trying different hyper parameters?

The best model showed an increase in training root mean square error (RMSE) from 48.086 to 55.72. However, the model's test error decreased significantly from 1.80405 to 0.6416. This suggests that the new model may have a slightly higher bias but better variance, resulting in improved generalization on the test data.

The reason for this improvement could be attributed to my focus on tree-based models with boosting ensemble techniques, specifically Gradient Boosting, CATboost, and XGBoost, during the hyperparameter optimization (HPO) phase. These models performed exceptionally well with their default settings. In contrast, before HPO, Autogluon attempted a wider range of model types, potentially leading to slight overfitting of the data.

By narrowing the focus to the tree-based models and utilizing boosting ensemble techniques, the new model achieved better generalization and reduced the test error, indicating improved performance and stronger ability to capture underlying patterns in the data.

### If you were given more time with this dataset, where do you think you would spend more time?
After assessing the results, I believe it would be beneficial to allocate more time to feature engineering and exploring new features. While hyperparameter tuning is crucial for achieving the best model performance, the significant improvement observed by solely adding the "hour" feature, without modifying the default settings of the models employed by Autogluon, suggests that feature engineering plays a crucial role in enhancing performance.

Therefore, I would prioritize dedicating additional time to feature engineering before delving into hyperparameter optimization. This approach would involve identifying and creating new features that could potentially provide valuable insights and boost the model's performance. By focusing on feature engineering initially, we can lay a solid foundation for the models to extract meaningful patterns and relationships from the data, potentially leading to further improvements in overall performance.

##Future Improvements
If given more time with this dataset, the following areas could be further explored and improved:

Feature Engineering: Allocate more time to feature engineering and explore new features that could provide valuable insights and boost the model's performance. Feature engineering plays a crucial role in enhancing model performance and extracting meaningful patterns from the data.

Hyperparameter Optimization: Continue experimenting with different hyperparameter settings to find the best configurations for each model. Fine-tuning the hyperparameters can lead to further improvements in model performance.
### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

model	hpo1	hpo2	hpo3	score
0	initial	default_vals	default_vals	default_vals	1.80405
1	add_features	default_vals	default_vals	default_vals	0.64160
2	hpo	GBM (Light gradient boosting) : num_boost_roun...	XGB (XGBoost): n_estimators : [lower=100, uppe...	CAT (CATBoost) : iterations : 100, depth : [lo...	0.54003

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
To summarize, this project highlighted the critical importance of both feature engineering and hyperparameter optimization in the machine learning workflow. It can be seen as an iterative process, where we alternate between extracting new features from the available data, performing exploratory data analysis (EDA), and experimenting with different models incorporating these new features. This iterative approach allows us to continuously refine the model's performance until we achieve satisfactory values for validation and test errors.

By iteratively extracting new features, conducting EDA, and evaluating different models, we can uncover valuable insights and patterns in the data. This process enables us to enhance the model's ability to capture the underlying relationships and make accurate predictions. It is through this iterative and cyclical approach that we can iteratively improve and refine our machine learning solution, ultimately aiming for optimal performance.

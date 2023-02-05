#Credit_Risk_Predictive_Model

This dataset contains 1000 entries with 10 variables. Each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The original dataset is prepared by Prof. Hofmann. 

Source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

The features are:

1. Age (numeric)

2. Sex (text: male, female)

3. Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

4. Housing (text: own, rent, or free)

5. Saving accounts (text - little, moderate, quite rich, rich)

6. Checking account (numeric, in DM - Deutsch Mark)

7. Credit amount (numeric, in DM)

8. Duration (numeric, in month)

9. Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

10. Risk (good, bad)

The process of building model:

(1)Performed data wrangling by fulfilling null values and conducted exploratory data analysis by plotting target and features’ distribution. Sensed that most clients loaned cars, radio/TV, and equipment.

(2)Transformed categorical data to numeric and split to train and test dataset. Built 7 machine learning models and compared to find GaussianNB had the highest fit score with 0.75  followed by xgboost.

(3)Tuned models’ parameters and confirmed GaussianNB was also the best model for predicting customers' credit quality, with an F1 score of 0.64.

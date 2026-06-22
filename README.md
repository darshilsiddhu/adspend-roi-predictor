# adspend-roi-predictor
A lot of D2C brands put money into multiple marketing channels every week — Google, Facebook, email, affiliates — and at the end of the month they're not really sure which one actually moved the needle. This project is an attempt to answer that question using real data and machine learning.

The app takes your weekly impression numbers across five channels and gives you two things: a predicted sales figure, and a breakdown of which channel is contributing the most to that number.


What it does


Predicts weekly sales based on marketing channel impressions
Shows each channel's actual contribution to sales (not just raw impressions)
Flags which channels are working and which ones aren't earning their budget
Runs as a live web app — no code knowledge needed to use it



How I built it

Dataset
Used a real media spend dataset with 3,051 weekly records covering five marketing channels and corresponding sales figures. The data spans 2018–2020 across multiple business divisions.

Model selection
I tested five different algorithms to find the best fit:

ModelCV R²Train-Test GapLinear Regression0.6530.145Ridge Regression0.6570.131Random Forest0.7440.286Gradient Boosting0.6970.250Bagging Regressor0.7400.280

Random Forest had the highest CV R² but also the worst overfitting gap. I went with Ridge Regression because the train-test gap is the smallest and the coefficients are directly interpretable — you can tell exactly how much each channel contributes per impression. For a business ROI tool, that interpretability matters more than squeezing out a few extra percentage points of accuracy.

Why Ridge over plain Linear Regression?
Ridge adds a regularization penalty that stops the model from over-relying on any single feature. It brought the overfitting gap down from 0.145 to 0.131 while keeping the CV R² essentially the same.


Tech stack


Python 3.10
scikit-learn (Ridge Regression, StandardScaler, Pipeline)
pandas, numpy
Streamlit (web app)
matplotlib (charts)
joblib (model serialization)



Running it locally

The app opens at http://localhost:8501


Project structure

adspend-roi-predictor/
│
├── data/
│   └── Sample_Media_Spend_Data.csv   # raw dataset
│
├── notebook/
│   └── analysis.ipynb                # EDA, model training, comparison
│
├── app.py                            # Streamlit web app
├── model.pkl                         # trained Ridge Regression pipeline
├── requirements.txt
└── README.md


Key findings


Facebook Impressions had the highest positive impact on sales per impression
Email Impressions showed consistent positive contribution across divisions
Affiliate Impressions showed a negative correlation with sales — likely because affiliate traffic tends to bring in lower-intent users in this dataset
The model explains about 65% of sales variation (CV R²: 0.657). The remaining 35% is likely driven by factors not in the dataset — pricing, competitor activity, seasonality, offline events



What I'd improve with more time


Add a budget optimizer: given a total weekly budget, what's the ideal split across channels?
Incorporate seasonality as a feature (quarter, holiday weeks)
Build a division-level model since the data has multiple divisions with potentially different channel dynamics
Add confidence intervals to the sales prediction so users know the uncertainty range



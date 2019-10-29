# Amazon Reviews - Prediction of Rating and Helpfulness (an NLP Use Case)

I have used Amazon Product Review dataset to train models that can predict sentiment expressed in a review as positive, negative or neural, and helpfulness of a review using the review text as a predictor. 

## Dataset

Amazon review dataset with 150,962,278 reviews is spread across 46 tsv files corresponding to 43 product categories. Amazon made this dataset available in amazon S3 bucket *amazon-reviews-pds*.

## Files contained in this project

- project_report: contains all the details about the project
- data_gathering_and_cleaning.ipynb: contains code for data gathering and cleaning
- exploratory_data_analysis.ipynb: contains code for exploratory data analysis
- ml_model_scaling.ipynb: contains the code for model scaling
- api.py: contains the code for API creation
- html_notebooks: this folder contains the html version of all the above four notebooks

## Serialized models 
The following folders are stored in S3 bucket s3://nlp-amazon-reviews-prasad/

- lr_model: this folder contains trained logistic regression model that has been serialized
- lrg_model: this folder contains trained linear regression model that has been serialized

## Resources used

For data gathering, cleaning, exploratory data analysis, ml prototyping and scaling Databricks platform [Databricks Community Edition](https://community.cloud.databricks.com) has been used. This platform has the pyspark environment set up. 



 


# Import required libraries
from flask import Flask, request, jsonify
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
import traceback
import pandas as pd
import numpy as np
import json

sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# API definition
app = Flask(__name__)

# Function to preidct rating
@app.route('/predict_rating', methods=['POST']) 
def predict_rating():

	json_ = request.json
	new_reviews = [{'review_body':review} for review in json_]
	new_reviews = [json.dumps(new_reviews)]
	# Convert new reviews to pyspark dataframe
	new_reviews_df = sc.parallelize(new_reviews)
	new_reviews_df = spark.read.json(new_reviews_df)
	
	# Tokenize user input
	tokenizer_usr_input = Tokenizer(inputCol="review_body", outputCol="tokens")
	# Extract doc2vec size of 300
	word2vec_usr_input = Word2Vec(vectorSize=300, minCount=0, inputCol="tokens", outputCol="features")
	doc2vec_pipeline_usr_input = Pipeline(stages=[tokenizer, word2vec])
	doc2vec_model_usr_input = doc2vec_pipeline.fit(new_reviews_df)
	doc2vec_new_reviews_df = doc2vec_model.transform(new_reviews_df)
	
	# Make predictions for new reviews
	prediction_df_lr = lr.transform(doc2vec_new_reviews_df).toPandas()
	# A function to convert numeric label to string label
	def convert_label(x):
	  if x == 0:
		return "Positive"
	  if x == 1:
		return "Negative"
	  return "Neutral"
		
	prediction_df_lr['prediction'] = prediction_df_lr['prediction'].apply(convert_label)
	prediction_df_lr.columns = ['Review Text', 'tokens', 'features', 'rawPrediction', 'probability', 'Predicted Rating']
	

	return json.dumps(dict(prediction_df_lr[['Review Text', 'Predicted Rating']]))

# Function to preidct helpfulness	
@app.route('/predict_helpfulness', methods=['POST']) 
def predict_helpfulness():

	json_ = request.json
	new_reviews = [{'review_body':review} for review in json_]
	new_reviews = [json.dumps(new_reviews)]
	# Convert new reviews to pyspark dataframe
	new_reviews_df = sc.parallelize(new_reviews)
	new_reviews_df = spark.read.json(new_reviews_df)
	
	# Tokenize user input
	tokenizer_usr_input = Tokenizer(inputCol="review_body", outputCol="tokens")
	# Extract doc2vec size of 300
	word2vec_usr_input = Word2Vec(vectorSize=300, minCount=0, inputCol="tokens", outputCol="features")
	doc2vec_pipeline_usr_input = Pipeline(stages=[tokenizer, word2vec])
	doc2vec_model_usr_input = doc2vec_pipeline.fit(new_reviews_df)
	doc2vec_new_reviews_df = doc2vec_model.transform(new_reviews_df)
	
	# Make predictions for new reviews
	prediction_df_lr = lrg.transform(doc2vec_new_reviews_df).toPandas()

	prediction_df_lr.columns = ['Review Text', 'tokens', 'Predicted Helpful Index']
	prediction_df_lrg['Predicted Helpful Index'] = round(prediction_df_lrg['Predicted Helpful Index'], 2)
	

	return json.dumps(dict(prediction_df_lrg[['Review Text', 'Predicted Helpful Index']]))

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8000 # If user doesn't provide any port the port will be set to 8000

    lr = LogisticRegressionModel.load('s3://nlp-amazon-reviews-prasad/lr_model') # Load the model stored as lr_model in S3 bucket
	lrg = LinearRegressionModel.load('s3://nlp-amazon-reviews-prasad/lr_model') # Load the model stored as lr_model in S3 bucket
    print ('Model loaded')

    app.run(port=port, debug=True)
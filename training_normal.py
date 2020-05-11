#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Usage:

    $ spark-submit training_normal.py hdfs:/user/ds6137/train_data_final.parquet hdfs:/user/ds6137/als_model_normal

'''


import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkConf, SparkContext

def main(spark, data_file, model_file):

    # Read data from parquet

    #training_data = 'hdfs:/user/ds6137/train_data_final.parquet'
    train_data = spark.read.parquet(data_file)
    print("read the training data")
    #train_data = spark.read.parquet('train_data_final.parquet')
    
    train_data.createOrReplaceTempView('train_data')

    als = ALS(maxIter=10, regParam = 0.1, rank = 100, userCol="user_id", itemCol="book_id", ratingCol="rating", coldStartStrategy="drop",implicitPrefs=True)
    # fitting the training data to the ALS model
    als_model_normal = als.fit(train_data)
    # saving the trained model
    als_model_normal.save(model_file)
    print("Model fitted and saved")

if __name__ == "__main__":
    print("Starting main")
    # Set parameters and create the spark session object
    spark = SparkSession.builder.appName("training_normal").master('yarn').config("spark.executor.memory", "15g").config("spark.driver.memory", "15g").config("spark.driver.maxResultsSize", "30G").getOrCreate()
    # spark = SparkSession.builder.appName("training_normal").getOrCreate()
    print("Spark session created")
    # training data file from command line
    data_file = sys.argv[1]

    # Get the model filename from the command line
    model_file = sys.argv[2]

    print("Calling main training function")
    # Call our main routine
    main(spark, data_file, model_file)
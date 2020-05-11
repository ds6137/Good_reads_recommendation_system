#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Usage:

    $ spark-submit testing_normal.py hdfs:/user/ds6137/als_model_normal hdfs:/user/ds6137/val_data_final.parquet hdfs:/user/ds6137/test_data_final.parquet
'''

import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row


def main(spark, model_file, val_file, test_file):
    test_data = spark.read.parquet(test_file)
    val_data = spark.read.parquet(val_file)
    als_model_normal = ALSModel.load(model_file)
    print("Imported trained model, validation and test data sets")

    # generating true values of book_id for each user_id from the validation set
    groundTruth_val = val_data.groupby("user_id").agg(F.collect_list("book_id").alias("val_truth"))
    print("Created ground truth df for validation set")

    # generating true values of book_id for each user_id from the test set
    groundTruth_test = test_data.groupby("user_id").agg(F.collect_list("book_id").alias("test_truth"))
    print("Created ground truth df for test set")

    groundTruth_val.createOrReplaceTempView('groundTruth_val')
    groundTruth_test.createOrReplaceTempView('groundTruth_test')

    # user_test_list=spark.sql('select distinct user_id from groundTruth_val where user_id=14')
    # rec = als_model_normal.recommendForUserSubset(user_test_list,500)

    #generating recs

    rec = als_model_normal.recommendForAllUsers(500)
    print("500 recommendations for all users generated")

    predictions_val = rec.join(groundTruth_val, rec.user_id == groundTruth_val.user_id, 'inner')

    predictions_test = rec.join(groundTruth_test, rec.user_id == groundTruth_test.user_id, 'inner')

    predAndLabels_val = predictions_val.select('recommendations.book_id', 'val_truth').rdd.map(tuple)
    predAndLabels_test = predictions_test.select('recommendations.book_id', 'test_truth').rdd.map(tuple)

    print("starting ranking metrics for validation data")

    metrics_val = RankingMetrics(predAndLabels_val)

    precision_val = metrics_val.precisionAt(500)
    map_val = metrics_val.meanAveragePrecision
    ndcg_val = metrics_val.ndcgAt(500)

    print('Validation set, Precision at 500: {}'.format(precision_val))
    print('Validation set, Mean Average Precision : {}'.format(map_val))
    print('Validation set, ndcgAt500 : {}'.format(ndcg_val))

    print("starting ranking metrics for test data")
    metrics_test = RankingMetrics(predAndLabels_test)

    precision_test = metrics_test.precisionAt(500)
    map_test = metrics_test.meanAveragePrecision
    ndcg_test = metrics_test.ndcgAt(500)

    print('Test set , Precision at 500: {}'.format(precision_test))
    print('Test set , Mean Average Precision : {}'.format(map_test))
    print('Test set, ndcgAt500 : {}'.format(ndcg_test))


# Only enter this block if we're in main
if __name__ == "__main__":
    print("starting main")
    # Create the spark session object
    spark = SparkSession.builder.appName('testing_normal').master('yarn').config("spark.executor.memory", "15g").config("spark.driver.memory", "15g").config("spark.driver.maxResultsSize", "30G").getOrCreate()
    # spark = SparkSession.builder.appName('testing_normal').master('yarn').getOrCreate()

    print("Spark session created")

    # Get the model file from the command line
    model_file = sys.argv[1]

    # Get the validation set from the command line
    val_file = sys.argv[2]

    # Get the test set from the command line
    test_file = sys.argv[3]

    # Call our main routine
    main(spark, model_file, val_file, test_file)

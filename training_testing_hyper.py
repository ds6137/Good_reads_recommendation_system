#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Usage:

    $ spark-submit training_testing_hyper.py hdfs:/user/ds6137/train_data_final.parquet hdfs:/user/ds6137/val_data_final.parquet hdfs:/user/ds6137/als_model_tuned
'''


import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics,RegressionMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row

def main(spark, data_file, val_file, model_file):
    train_data = spark.read.parquet(data_file)
    print("read the training data")
    train_data.createOrReplaceTempView('train_data')

    val_data = spark.read.parquet(val_file)
    print("read the validation data")
    val_data.createOrReplaceTempView('val_data')

    # generating true values of book_id for each user_id
    groundTruth_val = val_data.groupby("user_id").agg(F.collect_list("book_id").alias("val_truth"))
    print("Created ground truth df for validation set")
    groundTruth_val.createOrReplaceTempView('groundTruth_val')

    #reg_param and rank ranges
    reg_param = [0.01, 0.1, 1, 10]
    Rank = [10,20,30,100]

    #initiating dictionary to record performance of each metric
    precisions = {}
    count = 0
    total = len(reg_param) * len(Rank)

    for i in reg_param:
        for j in Rank:
            print(f"regParam: {i}, Rank: {j}")
            als = ALS(maxIter=10, regParam=i, rank=j, userCol="user_id", itemCol="book_id", ratingCol="rating",coldStartStrategy="drop", implicitPrefs=True)
            als_model = als.fit(train_data)

            # generating recs
            rec = als_model.recommendForAllUsers(500)
            # creating dataframe to have both true values and predicted values
            predictions_val = rec.join(groundTruth_val, rec.user_id == groundTruth_val.user_id, 'inner')
            # coverting to rdd for RankingMetrics()
            predAndLabels_val= predictions_val.select('recommendations.book_id','val_truth').rdd.map(tuple).repartition(1000)

            metrics_val = RankingMetrics(predAndLabels_val)
            # calculating metrics
            precision_val = metrics_val.precisionAt(500)
            map_val = metrics_val.meanAveragePrecision
            ndcg_val = metrics_val.ndcgAt(500)

            #storing the respective values in the dictionary with the MAP value as the key
            precisions[map_val] = [precision_val, ndcg_val, als_model, als]
            count += 1

            print(f"precision at: {precision_val}, MAP: {map_val}, NDCG: {ndcg_val}")

            print(f"finished {count} of {total}")


    #finding the best MAP and corresponding values
    best_map = max(list(precisions.keys()))
    best_precision,best_ndcg, best_model, best_ALS = precisions[best_map]
    best_model.write().overwrite().save(model_file)

    print(f"best MAP: {best_map}, with precision: {best_precision}, with NDCG: {best_ndcg},  regParam: {best_ALS.getRegParam},rank: {best_ALS.getRank}")



if __name__ == "__main__":
    print("starting main")
    # Create the spark session object
    spark = SparkSession.builder.appName('training_testing_hyper').master('yarn').config("spark.executor.memory", "15g").config("spark.driver.memory", "15g").config("spark.driver.maxResultsSize", "30G").getOrCreate()
    #spark = SparkSession.builder.appName('testing_normal').master('yarn').getOrCreate()

    print("Spark session created")
    # Get the data file from the command line
    data_file = sys.argv[1]
    #Get the validation set from the command line
    val_file = sys.argv[2]
    #Get the model file and path to be set from the command line
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, data_file, val_file, model_file)




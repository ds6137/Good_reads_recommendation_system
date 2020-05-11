#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Usage:
    $ spark-submit extension.py hdfs:/user/ds6137/test_data_final.parquet hdfs:/user/ds6137/als_model_tuned
'''

#We need sys to get the command line arguments
import sys

#And pyspark.sql to get the spark session
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row
from annoy import AnnoyIndex
from time import time

def main(spark, sc, test_file, model_file):
    test_data = spark.read.parquet(test_file)
    test_data.createOrReplaceTempView('test_data')

    test_users = spark.sql("select distinct user_id from test_data limit 800")
    #test_users = test_data.select("user_id").distinct().alias("user_id")
    groundTruth_test = test_data.groupby("user_id").agg(F.collect_list("book_id").alias("test_truth"))

    als_model = ALSModel.load(model_file)

    brute_force(als_model, groundTruth_test, test_users)

    trees = [10, 20, 40, 50]
    ks = [-1, 10, 50, 100]

    #annoy(alsmodel, groundTruth, testUsers, sc)

    for t in trees:
        for k in ks:
            annoy_model(als_model,sc, groundTruth_test, test_users, n_trees=t, search_k=k)

    print("finished!")

def brute_force(als_model, groundTruth_test, test_users):
    print("Normal Recommender system-Brute force")
    start_time = time()
    rec = als_model.recommendForUserSubset(test_users, 500)
    print("Normal-500 recommendations for test users generated")

    predictions_test = rec.join(groundTruth_test, rec.user_id == groundTruth_test.user_id, 'inner')

    predAndLabels_test = predictions_test.select('recommendations.book_id', 'test_truth').rdd.map(tuple)
    metrics_test = RankingMetrics(predAndLabels_test)
    precision_test = metrics_test.precisionAt(500)
    map_test = metrics_test.meanAveragePrecision
    print(f"Time taken: {time() - start_time}s")
    print(f"Precision at 500: {precision_test}")
    print(f"Mean Average Precision: {map_test}")



def annoy_model(als_model,sc, groundTruth_test, test_users, n_trees=10, search_k=-1):
    print(f"annoy model with n_trees: {n_trees}, search_k: {search_k}")

    sc = SparkContext.getOrCreate()

    user_factors = als_model.userFactors
    size = user_factors.limit(1).select(F.size("features").alias("calc_size")).collect()[0].calc_size
    start_time = time()
    index_size = AnnoyIndex(size)

    for row in user_factors.collect():
        index_size.add_item(row.id, row.features)

    index_size.build(n_trees)
    index_size.save("./annoy_result/annoy_t" + str(n_trees) + "_k_" + str(search_k) + ".ann")

    rec_list = [(user.user_id, index_size.get_nns_by_item(int(user.user_id), 500)) for user in test_users.collect()]

    temp = sc.parallelize(rec_list)

    print("Annoy-Recommendations (500) created for test users")

    rec = spark.createDataFrame(temp, ["user_id", "recommendations"])

    pred_test = rec.join(groundTruth_test, rec.user_id == groundTruth_test.user_id, 'inner')

    predAndLabels_test_annoy=pred_test.select('recommendations', 'test_truth').rdd.map(tuple)

    metrics_test_annoy = RankingMetrics(predAndLabels_test_annoy)
    precision_test_annoy = metrics_test_annoy.precisionAt(500)
    map_test_annoy = metrics_test_annoy.meanAveragePrecision

    print(f"Time taken: {time() - start_time}s")
    print(f"Precision at 500: {precision_test_annoy}")
    print(f"Mean Average Precision: {map_test_annoy}")

    index_size.unload()

if __name__ == "__main__":
    print("starting main")
    # Create the spark session object
    #spark = SparkSession.builder.appName('extension').master('yarn').config("spark.executor.memory", "15g").config("spark.driver.memory", "15g").config("spark.driver.maxResultsSize", "30G").getOrCreate()
    spark = SparkSession.builder.appName('testing_normal').getOrCreate()
    print("Spark session created")

    sc=SparkContext.getOrCreate()
    print("Spark context created")
    # Get the test data set file from the command line
    test_file = sys.argv[1]

    #Get the best model file and path to be set from the command line
    model_file = sys.argv[2]

    # Call our main routine
    main(spark,sc, test_file, model_file)
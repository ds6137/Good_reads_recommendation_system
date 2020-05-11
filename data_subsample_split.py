#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Usage:

    $ spark-submit data_subsample_split.py 0.01
'''

import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import RankingMetrics,RegressionMetrics
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.functions import row_number
from pyspark.sql import Window

def main(spark, subsample_fraction):
    interactions = spark.read.csv(f'hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header=True, inferSchema=True)
    books = spark.read.csv(f'hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header=True, inferSchema=True)
    users = spark.read.csv(f'hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header=True, inferSchema=True)

    print("Read complete interactions, users and books data")

    interactions.createOrReplaceTempView('interactions')
    books.createOrReplaceTempView('books')
    users.createOrReplaceTempView('users')

    #generating user list and identifying those user_id's which have at least 10 interactions in the whole dataset.
    clean_user_list = spark.sql('Select user_id from interactions group by user_id having count(book_id)>=10')
    print("generated user list that have at least 10 interactions")

    #sampling users based on the required fractions
    u_subsample = clean_user_list.sample(False, float(subsample_fraction), seed=42).cache()
    u_subsample.createOrReplaceTempView('u_subsample')

    print("Sampled users")

    #selecting all the interactions for the samopled users
    inter_subsample = spark.sql('select * from interactions where user_id in (select user_id from u_subsample)')
    inter_subsample.createOrReplaceTempView('inter_subsample')

    # generating train, val and test splits for users (from users dataset)
    users_train, users_val, users_test = u_subsample.randomSplit(weights=[.6, .2, .2], seed=42)
    print("generated train, val and test split on user data")
    users_train.createOrReplaceTempView('users_train')
    users_val.createOrReplaceTempView('users_val')
    users_test.createOrReplaceTempView('users_test')


    # selecting all the interactions for those users that are part of the training set
    inter_train = spark.sql('select * from inter_subsample where user_id in (select user_id from users_train)')
    inter_train.createOrReplaceTempView('inter_train')

    # selecting all the interactions for those users that are part of the val set
    inter_val = spark.sql('select * from inter_subsample where user_id in (select user_id from users_val)')
    inter_val.createOrReplaceTempView('inter_val')

    # selecting all the interactions for those users that are part of the test set
    inter_test = spark.sql('select * from inter_subsample where user_id in (select user_id from users_test)')
    inter_test.createOrReplaceTempView('inter_test')
    print("generated train, val and test split on interactions data based on user split")

    # assiging index to the validation interactions after sorting the unindexed interactions by user_id
    window = Window.orderBy(inter_val['user_id'])
    inter_val_new = inter_val.withColumn('row_number', row_number().over(window))
    inter_val_new.createOrReplaceTempView('inter_val_new')

    # filtering alternate validation interactions and adding the even indexed to the training dataset and odd indexed to the final validation set
    inter_val_train = inter_val_new.filter(inter_val_new.row_number % 2 == 0).select('user_id', 'book_id', 'is_read','rating', 'is_reviewed')
    inter_val_validation = inter_val_new.filter(inter_val_new.row_number % 2 != 0).select('user_id', 'book_id','is_read', 'rating','is_reviewed')

    inter_val_train.createOrReplaceTempView('inter_val_train')
    inter_val_validation.createOrReplaceTempView('inter_val_validation')

    # assiging index to the test interactions after sorting the unindexed interactions by user_id
    window2 = Window.orderBy(inter_test['user_id'])
    inter_test_new = inter_test.withColumn('row_number', row_number().over(window2))
    inter_test_new.createOrReplaceTempView('inter_test_new')

    # filtering alternate test interactions and adding the even indexed to the training dataset and odd indexed to the final test set
    inter_test_train = inter_test_new.filter(inter_test_new.row_number % 2 == 0).select('user_id', 'book_id', 'is_read','rating', 'is_reviewed')
    inter_test_test = inter_test_new.filter(inter_test_new.row_number % 2 != 0).select('user_id', 'book_id', 'is_read','rating', 'is_reviewed')

    inter_test_train.createOrReplaceTempView('inter_test_train')
    inter_test_test.createOrReplaceTempView('inter_test_test')

    #adding interactions from test and validation set to the training data
    test_t = inter_train.union(inter_val_train)
    test_s = test_t.union(inter_test_train)

    train_data_final = test_s
    print("Created final training dataset")

    #saving all the final datasets
    train_data_final.repartition(1).write.parquet('train_data_final_test.parquet')
    inter_val_validation.repartition(1).write.parquet('val_data_final_test.parquet')
    inter_test_test.repartition(1).write.parquet('test_data_final_test.parquet')

    print("saved all datasets in parquet format")

if __name__ == '__main__':
    print("starting main")
    # Create the spark session object
    spark = SparkSession.builder.appName('data_subsample_split').master('yarn').config("spark.executor.memory","15g").config("spark.driver.memory", "15g").config("spark.driver.maxResultsSize", "30G").getOrCreate()

    print("Spark session created")

    # Get the fraction size from the command line
    subsample_fraction = sys.argv[1]

    # Call our main routine
    main(spark, subsample_fraction)

import sys
import re
import datetime
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from pyspark import *
from pyspark.sql import *
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, ArrayType, StructType, StructField, FloatType, DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer,  OneHotEncoder,  StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression,  RandomForestRegressor, GBTRegressor
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import SparseVector, DenseVector


if __name__ == "__main__":
    spark = SparkSession.builder.appName("RFtraining").getOrCreate()
    #spark.conf.set("spark.executor.memory", '8g')
    train_data_path = sys.argv[1]
    data = spark.read.format('csv').option('header', 'true').option('inferSchema',
                                                                    'true').option('numPartitions',500).load(train_data_path)
    data_processed = data.withColumn('status', data.status.cast(StringType())).withColumn(
        'label', data.signal.cast(FloatType())).drop('signal','provider')
    train_df, test_df =data_processed.randomSplit([0.8,0.2], seed=777)
    ## one hot 编码
    categoricalCols = [field for (field, dataType) in data_processed.dtypes if dataType == "string"]
    inputOutputCols = [x+"index" for x in categoricalCols]
    oheOutputCols = [x+"OHE" for x in categoricalCols]
    stringIndexer = StringIndexer(inputCols=categoricalCols,
                                  outputCols=inputOutputCols,
                                  handleInvalid="skip")
    oheEncoder = OneHotEncoder(inputCols=inputOutputCols, outputCols=oheOutputCols)
    numeric_cols = [field for (field, dataType) in data_processed.dtypes if ((dataType != "string") & (field !='label'))]
    assembled_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="features_numeric")
    scaler = StandardScaler(inputCol="features_numeric", outputCol="features_scaled", withStd=True, withMean=True)
    assembled_inputs = oheOutputCols+["features_scaled"]


    ##将数据集组合成 feature向量
    vecAssembler = VectorAssembler(inputCols=assembled_inputs, outputCol='features')

    ##选择loss func
    evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')
    
    ##validationsplit manually
    stepSize= [0.5, 0.1, 0.05]
    maxDepth = [5, 10, 15, 20]
    maxIter= [10, 15, 20]
    


    for i in stepSize:
        for j in maxDepth:
              for k in maxIter:
                    #rf = RandomForestRegressor(featuresCol='features', labelCol='label',numTrees=i, maxDepth=j)
                    gbt = GBTRegressor(stepSize=i, maxDepth=j, maxIter=k)
                    pipeline_onetime = Pipeline(stages=[stringIndexer, oheEncoder, assembled_numeric,scaler, vecAssembler, gbt])
                    onetime_model = pipeline_onetime.fit(train_df)
                    rmse_train = evaluator.evaluate(onetime_model.transform(train_df), {evaluator.metricName: "rmse"})
                    rmse_test = evaluator.evaluate(onetime_model.transform(test_df), {evaluator.metricName: "rmse"})
                    print("stepsize={}, maxdepth={},maxIter={}, rmse_train={}, rmse_test={}".format(i, j,k,
                                                                                                    rmse_train, rmse_test))
        
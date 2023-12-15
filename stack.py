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
    spark = SparkSession.builder.appName("Stacking").getOrCreate()
    #spark.conf.set("spark.executor.memory", '8g')
    train_data_path = sys.argv[1]
    data = spark.read.format('csv').option('header', 'true').option('inferSchema',
                                                                    'true').option('numPartitions',500).load(train_data_path)
    data_processed = data.withColumn('status', data.status.cast(StringType())).withColumn('label',
        data.signal.cast(FloatType())).drop('signal','provider')
    train_df, val_df, test_df =data_processed.randomSplit([0.7,0.2, 0.1], seed=777)
    ## one hot 编码
    categoricalCols = [field for (field, dataType) in data_processed.dtypes if dataType == "string"]
    inputOutputCols = [x+"index" for x in categoricalCols]
    oheOutputCols = [x+"OHE" for x in categoricalCols]
    stringIndexer = StringIndexer(inputCols=categoricalCols,
                                 outputCols=inputOutputCols,
                                 handleInvalid="skip")
    oheEncoder = OneHotEncoder(inputCols=inputOutputCols,
                              outputCols=oheOutputCols)
    numeric_cols = [field for (field, dataType) in data_processed.dtypes if ((dataType != "string") & (field !='label'))]
    assembled_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="features_numeric")
    scaler = StandardScaler(inputCol="features_numeric", outputCol="features_scaled", withStd=True, withMean=True)
    assembled_inputs = oheOutputCols+["features_scaled"]

    ##将数据集组合成 feature向量
    vecAssembler = VectorAssembler(inputCols=assembled_inputs, outputCol='features')
    
    ##三种模型
    lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='predict_lr',
                          maxIter=100, regParam=0.002, elasticNetParam=0.0)
    rf = RandomForestRegressor(featuresCol='features', labelCol='label', predictionCol='predict_rf',
                               numTrees=22, maxDepth=15)
    gbt = GBTRegressor(featuresCol="features",labelCol='label', predictionCol='predict_gbt',
                      stepSize=0.5, maxDepth=10,maxIter=20,)
    pipeline_final = Pipeline(stages=[stringIndexer, oheEncoder, assembled_numeric, scaler, vecAssembler, lr, rf, gbt])
    
    
    ##fit 
    pipeline_model = pipeline_final.fit(train_df)
    test_fitted = pipeline_model.transform(test_df).select('label', 'predict_lr','predict_rf','predict_gbt')
    rmse_lr = RegressionEvaluator(labelCol="label", predictionCol='predict_lr', metricName="rmse")
    rmse_rf = RegressionEvaluator(labelCol="label", predictionCol='predict_rf', metricName="rmse")
    rmse_gbt = RegressionEvaluator(labelCol="label", predictionCol='predict_gbt', metricName="rmse")
    
    ##output RMSE for each model
    print('linear regression rmse==', rmse_lr.evaluate(test_fitted))
    print('random forest rmse==', rmse_rf.evaluate(test_fitted))
    print('gradient boost tree rmse==', rmse_gbt.evaluate(test_fitted))
    
    
    ###stacking
    vecAssembler_agg = VectorAssembler(inputCols=['predict_gbt','predict_rf','predict_lr'], outputCol='features_agg')
    lr_stack = LinearRegression(featuresCol='features_agg',labelCol='label' ,predictionCol='predict_stack')    
    
    
    ##fit stack model
    pipeline_stack_v2 =  Pipeline(stages=[vecAssembler_agg, lr_stack])
    val_fitted = pipeline_model.transform(val_df).select('label', 'predict_lr','predict_rf','predict_gbt')
    stack_model_v2 = pipeline_stack_v2.fit(val_fitted)
    
    
    
    ##evaluation
    rmse_stack = RegressionEvaluator(labelCol="label", predictionCol='predict_stack', metricName="rmse")
    print('rmse for val_data = ', rmse_stack.evaluate(stack_model_v2.transform(val_fitted)))
    print('rmse for test_data = ', rmse_stack.evaluate(
        stack_model_v2.transform(
            pipeline_model.transform(test_df).select('label', 'predict_lr','predict_rf','predict_gbt'))))
    
    
    

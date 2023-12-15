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
from pyspark.ml.feature import VectorAssembler, StringIndexer,  OneHotEncoder, StandardScaler
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
    data_processed = data.withColumn('status', data.status.cast(StringType())).withColumn('label',
        data.signal.cast(FloatType())).drop('signal','provider')
    #train_df, test_df = data_processed.randomSplit([0.8, 0.2], seed=777)
    
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

    ##选择模型
    #rf = RandomForestRegressor(featuresCol='features', labelCol='label')
    gbt = GBTRegressor(featuresCol="features",labelCol='label')

    ##选择loss func
    evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction')

    ##定义超参数的范围
    param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [9, 10, 11]) \
        .addGrid(gbt.stepSize, [0.5, 0.1, 0.05]) \
        .build()

    ##利用验证集调参
    tvs = TrainValidationSplit(estimator=gbt,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)
    
    pipeline = Pipeline(stages=[stringIndexer, oheEncoder, assembled_numeric,scaler, vecAssembler, tvs])
    pipelinemodel = pipeline.fit(data_processed)
    print('bestparams====', pipelinemodel.stages[5].bestModel.explainParams())
    ##预测
    pipelinemodel.transform(test_df).select('prediction','label').show()
    rmse = evaluator.evaluate(pipelinemodel.transform(train_df), {evaluator.metricName: "rmse"})
    print("train rmse==", rmse)
    rmse = evaluator.evaluate(pipelinemodel.transform(test_df), {evaluator.metricName: "rmse"})
    print("test rmse==", rmse)
    
  
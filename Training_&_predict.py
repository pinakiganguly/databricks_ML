# Databricks notebook source
df = spark.read.parquet("dbfs:/FileStore/Pinaki/ML_datasest/refined_data.parquet")
num_list = []
for col in df.dtypes:
    if col[1] == "int":
        num_list.append(col[0])
print("Integer variables:", len(num_list))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
inp_cols=df.columns
featureassembler=VectorAssembler(inputCols=num_list[1:], outputCol='Independent Features')

output_df=featureassembler.transform(df)
final_df=output_df.select('Independent Features','Saleprice')

# COMMAND ----------

train_X, test_X = final_df.randomSplit([0.8, 0.2])

# COMMAND ----------

train_X.limit(10).display()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

reg = LinearRegression(featuresCol="Independent Features", labelCol="Saleprice")
reg = reg.fit(train_X)

# COMMAND ----------

reg.coefficients

# COMMAND ----------

reg.intercept

# COMMAND ----------

pred=reg.evaluate(test_X)

# COMMAND ----------

predicted_df = pred.predictions.withColumnRenamed(
    "Saleprice", "Actual_Saleprice"
).withColumnRenamed("prediction", "Predicted_Saleprice")
pred.meanAbsoluteError, pred.meanSquaredError

# COMMAND ----------

predict_spark_df = predicted_df.select("Actual_SalePrice", "Predicted_SalePrice")
predict_pandas_df = predict_spark_df.toPandas()

# COMMAND ----------

from sklearn.metrics import r2_score

r2 = r2_score(
    predict_pandas_df["Actual_SalePrice"], predict_pandas_df["Predicted_SalePrice"]
)

# COMMAND ----------

print("R2 score:", r2*100,'%')

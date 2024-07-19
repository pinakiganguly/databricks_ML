# Databricks notebook source
# MAGIC %md
# MAGIC ###Reading the refined dataset from the dbfs & then extracting only the columns with numerical values

# COMMAND ----------

df = spark.read.parquet("dbfs:/FileStore/Pinaki/ML_datasest/refined_data.parquet")
num_list = []
for col in df.dtypes:
    if col[1] == "int":
        num_list.append(col[0])
print("Integer variables:", len(num_list))

# COMMAND ----------

df.printSchema() #printing the schema of the datatset

# COMMAND ----------

# MAGIC %md
# MAGIC ###Vectorizing the numerical columns and taking the saleprice column as target feature to train the model to identify the sale price depending on the Independent feature variables

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
inp_cols=df.columns
featureassembler=VectorAssembler(inputCols=num_list[1:], outputCol='Independent Features')

output_df=featureassembler.transform(df)
final_df=output_df.select('Independent Features','Saleprice')

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train test split with 80% training data and 20% testing data for the model

# COMMAND ----------

train_X, test_X = final_df.randomSplit([0.8, 0.2])

# COMMAND ----------

train_X.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Linear regression model imported and the mode is trained with the data

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

reg = LinearRegression(featuresCol="Independent Features", labelCol="Saleprice")
reg = reg.fit(train_X)

# COMMAND ----------

reg.coefficients

# COMMAND ----------

reg.intercept

# COMMAND ----------

# MAGIC %md
# MAGIC ###Evaluating the model by predicting the saleprice of the houses on test data

# COMMAND ----------

pred=reg.evaluate(test_X)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Renaming the columns of the resulting data frame for better understanding

# COMMAND ----------

predicted_df = pred.predictions.withColumnRenamed(
    "Saleprice", "Actual_Saleprice"
).withColumnRenamed("prediction", "Predicted_Saleprice")
pred.meanAbsoluteError, pred.meanSquaredError

# COMMAND ----------

# MAGIC %md
# MAGIC ###Converting spark  dataframe to pandas data frame for calculating the accuracy.

# COMMAND ----------

predict_spark_df = predicted_df.select("Actual_SalePrice", "Predicted_SalePrice")
predict_pandas_df = predict_spark_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Importing r2_score to calculate the accuracy of the model

# COMMAND ----------

from sklearn.metrics import r2_score

r2 = r2_score(
    predict_pandas_df["Actual_SalePrice"], predict_pandas_df["Predicted_SalePrice"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accuracy converted into %age

# COMMAND ----------

print("R2 score:", r2*100,'%')

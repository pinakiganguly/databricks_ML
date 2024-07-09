# Databricks notebook source
# MAGIC %run /Workspace/Users/pinaki.a.ganguly@accenture.com/databricks_ML/Data_ingestion_and_EDA

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
print(str_lis,"\n",len(str_lis))

# COMMAND ----------

df_ohe=df.select("*")
df_pandas = df_ohe.toPandas()
OH_encoder = OneHotEncoder(sparse=False)
OH_cols=pd.DataFrame(OH_encoder.fit_transform(df_pandas[str_lis]))
OH_cols.index = df_pandas.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = df_pandas.drop(str_lis, axis=1)
df_final = pd.concat([df_pandas,df_final, OH_cols], axis=1)
df_final_1=df_final.T.drop_duplicates().T

# COMMAND ----------

df_final_1

# COMMAND ----------

df_spark=spark.createDataFrame(df_final_1)

# COMMAND ----------

df_spark.display()

# COMMAND ----------



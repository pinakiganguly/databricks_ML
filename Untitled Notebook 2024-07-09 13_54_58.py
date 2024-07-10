# Databricks notebook source
import pandas as pd
df_pandas=pd.read_csv('/Workspace/Users/pinaki.a.ganguly@accenture.com/databricks_ML/houseprice.csv')

# COMMAND ----------

df_pandas.shape


# COMMAND ----------

df_pandas.head(5)

# COMMAND ----------

obj = (df_pandas.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

# COMMAND ----------

int_ = (df_pandas.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

# COMMAND ----------

print(num_cols)

# COMMAND ----------

fl = (df_pandas.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# COMMAND ----------

print(fl_cols)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.heatmap(df_pandas.corr(),
			cmap = 'BrBG',
			fmt = '.2f',
			linewidths = 2,
			annot = True)

# COMMAND ----------

unique_values = []
for col in object_cols:
  unique_values.append(df_pandas[col].unique().size)

# COMMAND ----------

print(unique_values)

# COMMAND ----------

s = (df_pandas.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(sparse=False)
OH_cols=pd.DataFrame(OH_encoder.fit_transform(df_pandas[object_cols]))
OH_cols.index = df_pandas.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = df_pandas.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# COMMAND ----------

df_final

# COMMAND ----------

type(object_cols)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# COMMAND ----------

X

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
X = df_spark.drop('SalePrice')
# Y = df_final['SalePrice']

# COMMAND ----------



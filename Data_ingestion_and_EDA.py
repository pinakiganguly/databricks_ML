# Databricks notebook source
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------


raw_df=spark.read.csv('dbfs:/FileStore/Pinaki/ML_datasest/houseprice.csv',header=True,inferSchema=True)

# COMMAND ----------

df1 = (
    raw_df.withColumn(
        "GarageYrBlt",
        when(raw_df.GarageYrBlt == "NA", "0.0").otherwise(raw_df.GarageYrBlt),
    )
    .withColumn("GarageYrBlt", col("GarageYrBlt").cast("float"))
    .withColumn(
        "LotFrontage",
        when(raw_df.LotFrontage == "NA", "0.0").otherwise(raw_df.LotFrontage),
    )
    .withColumn("LotFrontage", col("LotFrontage").cast("float"))
    .withColumn(
        "MasVnrArea",
        when(raw_df.MasVnrArea == "NA", "0.0").otherwise(raw_df.MasVnrArea),
    )
    .withColumn("MasVnrArea", col("MasVnrArea").cast("float"))
)

# COMMAND ----------

df1.display()

# COMMAND ----------

print(df1.count(),',',len(df1.columns))

# COMMAND ----------

str_lis = []
for col in df1.dtypes:
    if col[1] == "string":
        str_lis.append(col[0])
# print("Categorical variables:",str_lis)
print("Categorical variables:", len(str_lis))

num_list = []
for col in df1.dtypes:
    if col[1] == "int":
        num_list.append(col[0])
print("Integer variables:", len(num_list))

fl_list = []
for col in df1.dtypes:
    if col[1] == "float":
        fl_list.append(col[0])
print("Float variables:", len(fl_list))

# COMMAND ----------

num_cols=[]
for i in num_list:
    num_cols.append(i)
for j in fl_list:
    num_cols.append(j)
print(len(num_cols))

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=num_cols, outputCol=vector_col)
df_vector = assembler.transform(df1).select(vector_col)

# COMMAND ----------

matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corrmatrix = matrix.toArray().tolist()
print(corrmatrix)

# COMMAND ----------

df_corr = spark.createDataFrame(corrmatrix,num_cols)
df_corr.display()

# COMMAND ----------

def plot_corr_matrix(correlations,attr,fig_no,fig_size,annot=True):
    fig=plt.figure(fig_no,figsize=fig_size)
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticklabels(['']+attr)
    ax.set_yticklabels(['']+attr)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()

plot_corr_matrix(corrmatrix, num_cols, 234,(12,12),annot=True)

# COMMAND ----------

unique_vals=[]
for col in str_lis:
    unique_vals.append(df1.select(col).distinct().count())
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=str_lis,y=unique_vals)

# COMMAND ----------

df=df1.drop('Id')

# COMMAND ----------

df.display()

# COMMAND ----------

from pyspark.sql.functions import *
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).display()

# COMMAND ----------

df.write.parquet('dbfs:/FileStore/Pinaki/ML_datasest/refined_data.parquet')

# COMMAND ----------



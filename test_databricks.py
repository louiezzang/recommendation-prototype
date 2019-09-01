from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .getOrCreate()

# spark.sql("set spark.databricks.service.address=https://southeastasia.azuredatabricks.net")
# spark.sql("set spark.databricks.service.token=dapi61891f1776b2e49870028f1283c71e99")
# spark.sql("set spark.databricks.service.clusterId=0823-031151-brass263")
# spark.sql("set spark.databricks.service.orgId=4527285898187759")
# spark.sql("set spark.databricks.service.port=15001")

print("Testing simple count")

# The Spark code will execute on the Azure Databricks cluster.
print(spark.range(100).count())
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("SparkCommerce") \
    .getOrCreate()

# Load transactional data
transactions = spark.read.csv("path_to_transaction_data.csv", header=True, inferSchema=True)

# Market Basket Analysis (MBA)
fpGrowth = FPGrowth(itemsCol="products", minSupport=0.05, minConfidence=0.3)
model = fpGrowth.fit(transactions)

# Display frequent itemsets
model.freqItemsets.show()

# Display association rules
model.associationRules.show()

# Customer Profiling: Clustering customers based on purchasing behavior
# Assuming we have customer features like age, income, etc., and their transactional data
customer_data = spark.read.csv("path_to_customer_data.csv", header=True, inferSchema=True)

# Prepare features for clustering
assembler = VectorAssembler(inputCols=["age", "income", ...], outputCol="features")
kmeans = KMeans(featuresCol="features", k=5)  # Assuming 5 clusters
pipeline = Pipeline(stages=[assembler, kmeans])

# Fit pipeline to data
model = pipeline.fit(customer_data)

# Get cluster assignments
clustered_data = model.transform(customer_data)

# Show cluster assignments
clustered_data.select("customerId", "prediction").show()

# Stop SparkSession
spark.stop()

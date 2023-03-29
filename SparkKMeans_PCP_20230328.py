from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

# ----------------------------------------------------------
# PCP 20230328
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ----------------------------------------------------------

K = 5
# K = 10

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

# Load the data; note I am normalizing it with scale() - very important!
data = sc.parallelize(scale(createClusteredData(100, K)))
# No Scaling
# data = sc.parallelize(createClusteredData(100, K))


# Build the model (cluster the data)
# clusters = KMeans.train(data, K, maxIterations=10,
#        runs=10, initializationMode="random")


# Build the model (cluster the data)
# clusters = KMeans.train(data, K, maxIterations=10, initializationMode="random")
clusters = KMeans.train(data, K, maxIterations=3, initializationMode="random")



# Print out the cluster assignments
resultRDD = data.map(lambda point: clusters.predict(point)).cache()

print("Counts by value:")
counts = resultRDD.countByValue()
print(counts)

print("Cluster assignments:")
results = resultRDD.collect()
print(results)


# PCP 20230328
# sdf = spark.sql("select * from default_qubole_airline_origin_destination limit 10")
# display(sdf)
# sdf = resultRDD.sql("select * from resultRDD limit 10")
# sc.display(sdf)



# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# --- Within Set Sum of Squared Errors WSSSE ---
# Things to try:
# What happens to WSSSE as you increase or decrease K? Why?
# Increase K, WSSSE falls / Decrease K, WSSSE increases
# An increase in K produces more data points within the rage we are observing.
# Produces a higher chance of the points being closer to the Centroid.

# What happens if you don't normalize the input data before clustering?
# Large variation of points around each Centroid / in each Cluster.
# Massive increase in WSSSE
# Some clusters having many outlier points in and around them.

# What happens if you change the maxIterations or runs parameters?
# WSSSE increases with a lower number of iterations, so model becomes less accurate.

# Reduce() : What does it do?
# A fancy way of saying, "I want you to add up everything in this RDD into one final result." 
# reduce() will take the entire RDD, two things at a time, and combine them together using whatever
# function you provide. The function I'm providing it above is 
# "take the two rows that I'm combining together and just add them up."
# PCP --> Add up all numbers in the set to one number.
#
# WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
# print("Within Set Sum of Squared Error = " + str(WSSSE))
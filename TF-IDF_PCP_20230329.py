# TF-IDF: Term Frequency / Inverse Document Frequency
# Search Algorithm
# Search engine for Wikipedia using Apache Spark in MLlib
# #https://sparkbyexamples.com/pyspark/pyspark-py4j-protocol-py4jerror-org-apache-spark-api-python-pythonutils-jvm/
# Sometimes after changing/upgrading the Spark version, you may get this error due to the version incompatible between pyspark version 
# and pyspark available at anaconda lib. In order to correct it do the following.
# Note: copy the specified folder from inside the zip files and make sure you have environment variables set right as mentioned in the beginning.
# Copy the py4j folder from C:\apps\opt\spark-3.0.0-bin-hadoop2.7\python\lib\py4j-0.10.9-src.zip\ to C:\Programdata\anaconda3\Lib\site-packages\.
# And, copy pyspark folder from C:\apps\opt\spark-3.0.0-bin-hadoop2.7\python\lib\pyspark.zip\ to C:\Programdata\anaconda3\Lib\site-packages\
# You may need to restart your console some times even your system in order to affect the environment variables.
# When I upgraded my Spark version, I was getting this error, and copying the folders specified here resolved my issue.
# -----------------------------------------------------------

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# ----------------------------------------------------------
# PCP 20230328
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ----------------------------------------------------------

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
# rawData = sc.textFile("e:/sundog-consult/Udemy/DataScience/subset-small.tsv")
rawData = sc.textFile("C:/Users/pcpow/OneDrive/Desktop/DataScience_Udemy_20230321/DataScience/DataScience-Python3/subset-small_PCP_20230329.tsv")


fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

#print("fields: " + str(fields))
#print("documents: " + str(documents))
# print(fields)
# print(documents)



# Store the document names for later:
documentNames = fields.map(lambda x: x[1])


# TF-IDF: Term Frequency / Inverse Document Frequency
# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# At this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value.

# Let's compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

# I happen to know that the article for "Abraham Lincoln" is in our data
# set, so let's search for "Gettysburg" (Lincoln gave a famous speech there):

# First, let's figure out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Now we will extract the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# We'll zip in the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames)

# And, print the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())

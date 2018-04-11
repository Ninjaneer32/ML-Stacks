# Machine Learning Languages, Frameworks and Stacks

I was searching for article on Machine Learning stacks but didn't find anything too relevant. I have tried to gather information and put it up here. Along with stacks, I have compiled list of languages and frameworks as well.

I spent quite a time in compiling this information but number of languages, frameworks and stacks are quite large in number, and hence I decided to publish this with whatever information I have so far. I am looking forward to everyone to add further data to article to make it as complete as possible.

## Overview

![ML Stack](https://github.com/kapilkathuria/ML-Stacks/blob/master/Stacks_3.png)

For creating & deploying machine learning, key stages are Problem Understanding, Data Understanding, Model Building, Model Evaluation and Model Deployment.

Once stated problem is understood, next things is understanding data - which is shown as Data Processing (2nd Stack). To understand data, data need to be somewhere (shown as Data Source, 1st Stack). For model building, depending on problem, either deep learning (3rd stack) or Normal Machine Learning Algorithm (4th stack) are used. For Model Evaluation - again Data Processing (2nd stack).

At lowest level in stack is hardware layer in Orange-rust color. Take a note that GPU is bold for deep learning. Deep Learning algorithms requires lot of maths and this is where GPU helps a lot. Most of libraries (Tensorflow, Pytorch etc.) mentioned under this stack are written to use Nividia GPU. TPU is Tensorflow Processing Unit (TPU) which is available only in Google Cloud.

2nd level from bottom is Platform level in Blue color. Bigdata platforms like Hadoop and Spark would source of data most of the times but source of data could be anything from csv file to any SQL / NOSQL database (MS SQL, Oracle, Mongodb, Postgres etc.).

3rd and 4th level is Software Stack. Various languages can be used like - C/C++/Java/Python/R/MatIab/Octave/ScaIa/Julia/Node/Go/Rust. Out of these Python/R/MatIab/Octave/ScaIa/Julia seems to be most popular. Python is most popular as of today, and that's covered in detail in picture above. Most common Frameworks / Libraries used in Python are shown. Please note, MLLIB library is for using only Spark. Keras library is shown at 4th level as it works on Tensorflow and Torch to simplify usage of underlying libraries.

## Languages

### C/C++

Low level language with excellent performance. Pro: Direct Access to GPU using Nvidia libraries CUDA/cuDNN. Cons: Huge code. Memory management etc. need to be handled. Used by: Used by very big firms like Google, Facebook etc. where hardcore highly optimized algorithms are written.

### R 

Math language for ML. It's not procedural language or Object oriented language. Pros: Optimized for mathematical operations. Very easy to handle matrices, vectors. Specialized for linear algebra, stats and calculus. Supports ML robustly. Number of lines are reduced drastically. Common on Data Analysis side than for Machine Learning. Preferred among Math Languages (R, Matlab, Octave, Julia).

### Matlab

Similar to R, it's another Math language used for ML. Popular in research and academia. Mainly in academia and research. Some people use this initially to find model and then model is migrated to more robust languages like Python / Java.Pros: Better for linear algebra than stats. Cons: This is licensed.

### Octave

Very similar to Matlab and is open source.

### Julia 

Similar to R, newer and slicker. Pros: Powerful ML models, syntactical sugar. Cons: Newer and not used widely as of now

### Java

JVM Language. Meant for Hadoop based pipeline. Mahout / MLLIB libraries for connecting to MPReduce / Spark. Pros: Enterprise, robust.

### Sacala

JVM based Functional Language. Meant for highly scalable data pipeline on Mapreduce / Spark. Pros: Dynamic, functional version of Java with Syntactical Sugar.

### Python

Most popular as of today. It has huge ecosystem - larger number of libraries for Machine Learning and it's complete language (i.e. web services, web pages can be created as well).

* NumPy → Math matrices, vector) / Matlab

* Panda → Stats. R equivalent

* SciKit Learn → ML Algo (Linear Reg, Log Reg, SVN, Baysian Interference)

* PySpark → Data Mining on Spark

* Other frameworks like Django, flask, sqlalchme, Tkinter/PySide

* Deep Learning: Theano / Torch / Tensorflow -- code converted to C

* Server (Flask, Django)

### Other Languages

SAS / Node / Go / Rust: these are not widely used in machine learning

### References

http://www.analyticbridge.datasciencecentral.com/profiles/blogs/r-python

http://www.kdnuggets.com/2017/01/most-popular-language-machine-learning-data-science.html

http://r4stats.com/2017/02/28/r-passes-sas/

https://www.analyticsvidhya.com/blog/2017/09/sas-vs-vs-python-tool-learn/



## Frameworks

### Python ML libraries 

Numpy, Pandas, scikit-learn (sklearn) - all of libraries together can be considered as Python Framework for Machine Learning. Pros: Widely used, and big community (for development & support)

### Theano

Oldest but not be actively developed in future (MILA will stop developing Theano). Used for Mathematical ops / computation graphs. API is quite low level. Works with GPUs and performs efficient symbolic differentiation. Blocks / Lasagne / Keras libraries used on top of Theano for deep learning

http://deeplearning.net/software/theano/

Pros: Flexible. Performant if used properly.

Cons: Substantial learning curve. Compiling complex symbolic graphs can be slow.

### Torch / PyTorch 

Created by Facebook. Specialized in image recognition using CNN (Convolution Neural Network). Written on Lua (Language) initially but recently released Python based API - PyTorch.

### TensorFlow

Used by all teams within Google. Most commonly used. Layers: Maths / Middle - ML Algo / High - Deep Learning. Keras: works with TensorFlow as well. Supports NViDIA GPU / CUDA / CUDNN

Pros: Code converted to C. Can run on Mobile. Can be compiled to run on CPU / GPU. Backed by software giant Google. Very large community. Low level and high level interfaces to network training. Faster model compilation than Theano-based options. Clean multi-GPU support. Provides utilities for efficient data pipelining, and has built-in modules for the inspection, visualization, and serialization of models. • Initially slower at many benchmarks than Theano-based options, although Tensorflow is catching up.

Cons: RNN support is still outclassed by Theano

### TensorForce

Deep reinforcement learning API built on top of TensorFlow

https://reinforce.io/blog/introduction-to-tensorforce/

### Core ML

For Apple Devices - works on iPhone as well.

### Others

Caffe (old-n-dying, C++)

CNTK (MS)

mxnet (Amazon)

OpenCV (vision only)

Mahout & DL4J: Deep Learning for Java

### References

http://www.kdnuggets.com/2017/02/python-deep-learning-frameworks-overview.html

## Platforms / Stacks

### Could vs Open Source Stack

Cloud: IBM, AWS, Azure etc. Better for regular solution like image recognition. Less Techy people can manage building models

In premise Stack: Apache Spark/ Mapreduce, MLLib with Python / R / Scala. Better for custom solution

### Cloud Stacks

#### Amazon Machine Learning

Use drag-n-drop machine learning tools for data preparation, feature engineering, algorithm selection and evaluation. Data must be in AWS. P2.xlarge (GPU server isnatnce on EC2)

#### IBM Watson / IBM Bluemix

Data must be in Blemix

#### Azure Machine Learning

Small data can be outside but if more than GB, it should be Azure. Free version with limited capabilities available for development & personal use

Pre-build algo: Boosted decision tree, Bayesian recommendation systems, Deep Neural networks, Decision jungles, Classiciation: Regression Clustering, Binaary & Multiclass

#### Google Machine Learning / GCP 

Data must be in Google cloud.

Pros: TPU (tensor processing unit): expected to major performance boost, Faster GPU (less cost)

Services: NLP APIs, Speech API, Translation API, Vision API, Text Analysis, Jobs API (Alpha), Video API (Beta)

#### Databricks

Cloud based Apache Spark. Company founded by creators of Apache Spark. Web-based platform for working with Spark.

#### Fujitsu

Fujitsu Cloud Service K5 Zinrai

#### Others

BigML / FICO / Dato / Datarobot / Weka

#### References

http://searchbusinessanalytics.techtarget.com/feature/Machine-learning-platforms-comparison-Amazon-Azure-Google-IBM

http://www.datasciencecentral.com/profiles/blogs/cloud-machine-learning-apis

http://www.hadoop360.datasciencecentral.com/blog/cloud-machine-learning-platforms-vs-apache-spark-solutions

### In Premise stacks

#### Open Source

Apache Spark / Apache Flink, MLLib with Python / R / Scala, Python ML Libraries, Tensorflow (& other Deep Learning Libraries), AIRI

AIRI: NVIDIA GPI Compute with Flash memory for high performance. Details here
See Apache Spark vs Apache Flink here: https://www.dezyre.com/article/apache-flink-vs-spark-will-one-overtake-the-other/282

#### Fujitsu

Fujitsu AI Solution Zinrai Deep Learning System

http://www.fujitsu.com/global/about/resources/news/press-releases/2017/0516-04.html

#### KNIME
https://www.knime.com/software
Provides the fully open-source KNIME Analytics Platform. It offers highly rated data
access and manipulation capabilities, a breadth of algorithms, and a comprehensive machine-learning toolbox suitable for both beginners and expert data scientists. KNIME's platform integrates with other tools and platforms, such as R, Python, Spark, H2O.ai, Weka, DL4J and Keras. 
Limitation: A KNIME Server deployment is currently limited to a single host. 

#### H2O
offers an open-source machine-learning platform.
H2O Flow: its core component; H2O Steam; H2O Sparkling Water: Spark integration; and H2O Deep Water: deep-learning capabilities.

### Nvidia vs AMD GPU

#### Nvidia

CUDA(Compute unified device Architecture) (parallel computing) and cuDNN (Nural network) libraries expose GPU to be used by other languages.

Cons: Costly, New MAC doesn't support Nvidia

#### AMD GPU

OpenCL: Equvaluent to CUDA.

Cons: Tensorflow doesn't support AMD. Not much used.

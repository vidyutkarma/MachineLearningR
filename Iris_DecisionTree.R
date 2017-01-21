# We shall use the C50 package.
if(!require("C50")) 
{
  install.packages("C50")
  library(C50)
}

if(!require("gmodels")) 
{
  install.packages("gmodels")
  library(gmodels)
}
#Lets set the working directory
setwd("/Users/vijitkarma/Desktop/Study/Books/MachineLearning/Independent")

# We shall read the csv file. Since, here only the label is string and we need it as nominal, we shall not use StringsAsFactors option

IrisDataset <- read.csv("Iris.csv")

# Lets see some of the records in the data set
head(IrisDataset) 

#It seems that is not randomly arranged. We shall arrange it randomly, so that we can extract validation and test set
# out of it. We shall use the runif in conjunction with order() to generate random frame. To use runif(), we need to know the 
# number of rows in the data frame. Also, we shall set the seed, so that if this is repeated, we shall get the same order

any(is.na(IrisDataset)) # To check if NAs are present in the data set. The output here was false.

set.seed(12345)
IrisRandom <- IrisDataset[order(runif(NROW(IrisDataset))), ]

# Lets check the summary to ensure that the pre-random and post-random datasets have the same properties

summary(IrisDataset)
summary(IrisRandom)

#Lets devide the data in train and validation set in the ratio 70:30

IrisTrain <- IrisRandom[1:105, -1]
IrisVal <- IrisRandom[106:150, -1]

# Lets check if we have proportionate species samples in our training and validation set.
prop.table(table(IrisTrain$Species))

# Iris-setosa Iris-versicolor  Iris-virginica 
# 0.3333333       0.3523810       0.3142857 

prop.table(table(IrisVal$Species))

# Iris-setosa Iris-versicolor  Iris-virginica 
# 0.3333333       0.2888889       0.3777778 

#Training the model
speciesClassifierModel <- C5.0(IrisTrain[, -5], IrisTrain$Species)

#Basic check of the tree

speciesClassifierModel

# Call:
#   C5.0.default(x = IrisTrain[, -5], y = IrisTrain$Species)
# 
# Classification Tree
# Number of samples: 105 
# Number of predictors: 4 
# 
# Tree size: 3 
# 
# Non-standard options: attempt to group attributes

#Checking the decisions

summary(speciesClassifierModel)

# Call:
#   C5.0.default(x = IrisTrain[, -5], y = IrisTrain$Species)
# 
# 
# C5.0 [Release 2.07 GPL Edition]  	Sat Jan 21 18:42:46 2017
# -------------------------------
#   
#   Class specified by attribute `outcome'
# 
# Read 105 cases (5 attributes) from undefined.data
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (37)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.7: Iris-versicolor (37/3)
# PetalWidthCm > 1.7: Iris-virginica (31/1)
# 
# 
# Evaluation on training data (105 cases):
# 
# Decision Tree   
# ----------------  
# Size      Errors  
# 
# 3    4( 3.8%)   <<
# 
# 
# (a)   (b)   (c)    <-classified as
# ----  ----  ----
# 37                (a): class Iris-setosa
# 34     1    (b): class Iris-versicolor
# 3    30    (c): class Iris-virginica
# 
# 
# Attribute usage:
# 
# 100.00%	PetalWidthCm
# 
# 
# Time: 0.0 secs

#As can be seen above, 4 training samples were incorrectly classified.

#Prediction

speciesPred <- predict.C5.0(speciesClassifierModel, IrisVal[, -5])

CrossTable(speciesPred, IrisVal[, 5], prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c("predicted", "actual"))

# Cell Contents
# |-------------------------|
#   |                       N |
#   |         N / Table Total |
#   |-------------------------|
#   
#   
#   Total Observations in Table:  45 
# 
# 
# | actual 
# predicted |     Iris-setosa | Iris-versicolor |  Iris-virginica |       Row Total | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-setosa |              12 |               0 |               0 |              12 | 
#   |           0.267 |           0.000 |           0.000 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-versicolor |               1 |              15 |               2 |              18 | 
#   |           0.022 |           0.333 |           0.044 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-virginica |               0 |               0 |              15 |              15 | 
#   |           0.000 |           0.000 |           0.333 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Column Total |              13 |              15 |              17 |              45 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   

#As can be seen above, the accuracy is 93.33%

#Applying boosting so that many decision trees are built and the trees vote for the best class for each example. 

speciesClassifierModelBoosted <- C5.0(IrisTrain[, -5], IrisTrain$Species, trials = 10)

summary(speciesClassifierModelBoosted)

# Call:
#   C5.0.default(x = IrisTrain[, -5], y = IrisTrain$Species, trials = 10)
# 
# 
# C5.0 [Release 2.07 GPL Edition]  	Sat Jan 21 18:57:44 2017
# -------------------------------
#   
#   Class specified by attribute `outcome'
# 
# Read 105 cases (5 attributes) from undefined.data
# 
# -----  Trial 0:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (37)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.7: Iris-versicolor (37/3)
# PetalWidthCm > 1.7: Iris-virginica (31/1)
# 
# -----  Trial 1:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (28.1)
# PetalWidthCm > 0.5:
# :...PetalLengthCm <= 4.9: Iris-versicolor (41.5/10.1)
# PetalLengthCm > 4.9: Iris-virginica (35.4/1.5)
# 
# -----  Trial 2:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (22)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.4: Iris-versicolor (13.7)
# PetalWidthCm > 1.4:
# :...PetalWidthCm > 1.8: Iris-virginica (16.7)
# PetalWidthCm <= 1.8:
# :...SepalWidthCm <= 3.1: Iris-virginica (45.9/11.5)
# SepalWidthCm > 3.1: Iris-versicolor (6.7)
# 
# -----  Trial 3:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (17.1)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.6: Iris-versicolor (46.3/8.6)
# PetalWidthCm > 1.6: Iris-virginica (41.6/10.3)
# 
# -----  Trial 4:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (13.8)
# PetalWidthCm > 0.5:
# :...PetalWidthCm > 1.8: Iris-virginica (10.5)
# PetalWidthCm <= 1.8:
# :...SepalLengthCm > 6.4: Iris-versicolor (16.3/0.4)
# SepalLengthCm <= 6.4:
# :...SepalWidthCm > 3.1: Iris-versicolor (9.3)
# SepalWidthCm <= 3.1:
# :...PetalLengthCm <= 4.4: Iris-versicolor (9)
# PetalLengthCm > 4.4: Iris-virginica (46.1/14.8)
# 
# -----  Trial 5:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (10.9)
# PetalWidthCm > 0.5:
# :...PetalWidthCm > 1.8: Iris-virginica (8.3)
# PetalWidthCm <= 1.8:
# :...SepalLengthCm <= 5.1: Iris-virginica (5.9/0.9)
# SepalLengthCm > 5.1:
# :...PetalLengthCm <= 4.7: Iris-versicolor (28.9)
# PetalLengthCm > 4.7:
# :...SepalWidthCm <= 2.3: Iris-virginica (6.7)
# SepalWidthCm > 2.3: Iris-versicolor (44.2/15.8)
# 
# -----  Trial 6:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (8.7)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.7: Iris-versicolor (63.8/17.7)
# PetalWidthCm > 1.7: Iris-virginica (32.5/5.4)
# 
# -----  Trial 7:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (7.2)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.4: Iris-versicolor (11.8)
# PetalWidthCm > 1.4:
# :...SepalWidthCm <= 2.8: Iris-virginica (49.8/11.9)
# SepalWidthCm > 2.8: Iris-versicolor (36.3/13.1)
# 
# -----  Trial 8:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (5.9)
# PetalWidthCm > 0.5:
# :...PetalWidthCm <= 1.4: Iris-versicolor (9.7)
# PetalWidthCm > 1.4:
# :...PetalLengthCm > 5.1: Iris-virginica (17.6)
# PetalLengthCm <= 5.1:
# :...SepalLengthCm <= 5.1: Iris-virginica (6.3)
# SepalLengthCm > 5.1:
# :...PetalLengthCm <= 4.7: Iris-versicolor (11.8)
# PetalLengthCm > 4.7:
# :...SepalWidthCm <= 3.1: Iris-virginica (45.4/14.2)
# SepalWidthCm > 3.1: Iris-versicolor (8.3/0.8)
# 
# -----  Trial 9:  -----
# 
# Decision tree:
# 
# PetalWidthCm <= 0.5: Iris-setosa (4.7)
# PetalWidthCm > 0.5:
# :...PetalLengthCm > 5.1: Iris-virginica (13.9)
# PetalLengthCm <= 5.1:
# :...PetalWidthCm > 1.8: Iris-virginica (6.3)
# PetalWidthCm <= 1.8:
# :...SepalLengthCm > 6.3: Iris-versicolor (14.8)
# SepalLengthCm <= 6.3:
# :...SepalWidthCm > 3.1: Iris-versicolor (6.2)
# SepalWidthCm <= 3.1:
# :...PetalWidthCm <= 1.6: Iris-versicolor (45.1/13.8)
# PetalWidthCm > 1.6: Iris-virginica (14)
# 
# 
# Evaluation on training data (105 cases):
# 
# Trial	    Decision Tree   
# -----	  ----------------  
# Size      Errors  
# 
# 0	     3    4( 3.8%)
# 1	     3    7( 6.7%)
# 2	     5    9( 8.6%)
# 3	     3    4( 3.8%)
# 4	     6    8( 7.6%)
# 5	     6   11(10.5%)
# 6	     3    4( 3.8%)
# 7	     4   23(21.9%)
# 8	     7    5( 4.8%)
# 9	     7    2( 1.9%)
# boost	          1( 1.0%)   <<
# 
# 
# (a)   (b)   (c)    <-classified as
# ----  ----  ----
# 37                (a): class Iris-setosa
# 34     1    (b): class Iris-versicolor
# 33    (c): class Iris-virginica
# 
# 
# Attribute usage:
# 
# 100.00%	PetalWidthCm
# 64.76%	PetalLengthCm
# 61.90%	SepalWidthCm
# 48.57%	SepalLengthCm
# 
# 
# Time: 0.0 secs

#As can be seen above, error has reduced on the training data. We can still increase the number of trials since the model seems like still improving

speciesPredBoosted <- predict.C5.0(speciesClassifierModelBoosted, IrisVal[, -5])

CrossTable(speciesPredBoosted, IrisVal[, 5], prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c("predicted", "actual"))

# Cell Contents
# |-------------------------|
#   |                       N |
#   |         N / Table Total |
#   |-------------------------|
#   
#   
#   Total Observations in Table:  45 
# 
# 
# | actual 
# predicted |     Iris-setosa | Iris-versicolor |  Iris-virginica |       Row Total | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-setosa |              12 |               0 |               0 |              12 | 
#   |           0.267 |           0.000 |           0.000 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-versicolor |               1 |              15 |               2 |              18 | 
#   |           0.022 |           0.333 |           0.044 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-virginica |               0 |               0 |              15 |              15 | 
#   |           0.000 |           0.000 |           0.333 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Column Total |              13 |              15 |              17 |              45 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   

#As can be seen above, the accuracy remained the same. If some misclassification is costlier than others then we can also implement costing
#wherein we can penalize the costlier misclassification. First we need to define a matrix which assigns the cost to the misclassification
#and then pass it as the "costs" parameter to the trainer.

error_cost <- matrix(c(0, 0, 0, 1, 0, 4, 0, 0, 0), nrow = 3, byrow = TRUE)

error_cost <- matrix(c(0, 0, 0, 1, 0, 4, 0, 0, 0), nrow = 3, byrow = TRUE)
error_cost
# [,1] [,2] [,3]
# [1,]    0    0    0
# [2,]    1    0    4
# [3,]    0    0    0

#Above, we have assigned more cost to the misclassification "Iris-virginica".

speciesClassifierModel <- C5.0(IrisTrain[, -5], IrisTrain$Species, costs = error_cost)

speciesPred <- predict.C5.0(speciesClassifierModel, IrisVal[, -5])

CrossTable(speciesPred, IrisVal[, 5], prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c("predicted", "actual"))

# Cell Contents
# |-------------------------|
#   |                       N |
#   |         N / Table Total |
#   |-------------------------|
#   
#   
#   Total Observations in Table:  45 
# 
# 
# | actual 
# predicted |     Iris-setosa | Iris-versicolor |  Iris-virginica |       Row Total | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-setosa |              12 |               0 |               0 |              12 | 
#   |           0.267 |           0.000 |           0.000 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-versicolor |               1 |              15 |               2 |              18 | 
#   |           0.022 |           0.333 |           0.044 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Iris-virginica |               0 |               0 |              15 |              15 | 
#   |           0.000 |           0.000 |           0.333 |                 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   Column Total |              13 |              15 |              17 |              45 | 
#   ----------------|-----------------|-----------------|-----------------|-----------------|
#   

#In this case, the penalizing did not help as there are still 2 misclassification of "Iris-viriginica"

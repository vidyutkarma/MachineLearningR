#Implementing concrete strength prediction using neural networks using the concrete.csv data from UCI site

# We shall use the following packages.
if(!require("tidyr")) 
{
  install.packages("tidyr")
  library(tidyr)
}

if(!require("ggplot2")) 
{
  install.packages("ggplot2")
  library(ggplot2)
}

if(!require("neuralnet")) 
{
  install.packages("neuralnet")
  library(neuralnet)
}

#Reading the data
concrete <- read.csv("concrete.csv")

#Checking the structure of data
str(concrete)
#'data.frame':	1030 obs. of  9 variables:
#$ cement      : num  540 540 332 332 199 ...
#$ slag        : num  0 0 142 142 132 ...
#$ ash         : num  0 0 0 0 0 0 0 0 0 0 ...
#$ water       : num  162 162 228 228 192 228 228 228 228 228 ...
#$ superplastic: num  2.5 2.5 0 0 0 0 0 0 0 0 ...
#$ coarseagg   : num  1040 1055 932 932 978 ...
#$ fineagg     : num  676 676 594 594 826 ...
#$ age         : int  28 28 270 365 360 90 365 28 28 28 ...
#$ strength    : num  80 61.9 40.3 41 44.3 ...

#As can be seen above, the data has 8 attributes with one with one outcome variabe "strength". 

#Lets check the min and max values

apply(concrete, MARGIN = 2, min)
#cement         slag          ash        water superplastic    coarseagg      fineagg          age     strength 
#102.00         0.00         0.00       121.80         0.00       801.00       594.00         1.00         2.33 

apply(concrete, MARGIN = 2, max)
#cement         slag          ash        water superplastic    coarseagg      fineagg          age     strength 
#540.0        359.4        200.1        247.0         32.2       1145.0        992.6        365.0         82.6 

#As can be seen the data is ranging from zero to more than hundred. Lets see how the data is distributed.

#create long data using gather() function from tidyr package. 

long.concrete <- gather(concrete, attribute, value, 1:9)

head(long.concrete)

# attribute value
# 1    cement 540.0
# 2    cement 540.0
# 3    cement 332.5
# 4    cement 332.5
# 5    cement 198.6
# 6    cement 266.0

#histogram using ggplot
ggplot(long.concrete, aes(x = value, ..density..)) + facet_wrap(~attribute, scales = "free_x") + geom_histogram()

#Distribution attached.
#As can be seen, the data is mostly non-normal. Hence, we shall use the normalize function (defined in the functions section of this document)

normalized.concrete <- as.data.frame(lapply(concrete, normalize))

#check the max and min value in the normalized data frame.
max(normalized.concrete)
#[1] 1

min(normalized.concrete)
#[1] 0

#Data has been normalized and the values now lie between 0 and 1

#Lets divide the data into training and test sets (70:30 ratio)
#Since the original data set is randomized, we shall straightaway partition the data

train.concrete <- normalized.concrete[1:721, ]
test.concrete <- normalized.concrete[722:1030, ]

#we shall use the neuralnet package in R to accomplish this task
# creating the model with a default single hidden neuron
model.concrete <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = train.concrete)

#We can check the structure of the neural network using plot(model_name)
plot(model.concrete)

#as can be seen, the plot gives the weight parameters, sum of squarred errors and the number of steps taken.

#Now let us test our model. We use compute() function for this. We can also use predict() for this. compute() returns a list with 
#two components. $neurons gives the number of neurons in each layer. $net.results gives the predicted values.

model.results <- compute(model.concrete, test.concrete[1:8])

predicted.strength <- model.results$net.result

#Lets check the correlation between the predicted results and the actual results
cor(predicted.strength, test.concrete$strength)
# [,1]
# [1,] 0.7505827775

#We got a correlation of 0.75 which is quite good considering we used only one hidden neuron. Let's try with more number of hidden neurons

model.concrete.enhanced <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = train.concrete, hidden = 5)
model.results.enhanced <- compute(model.concrete.enhanced, test.concrete[1:8])

predicted.strength.enhanced <- model.results.enhanced$net.result

cor(predicted.strength.enhanced, test.concrete$strength)
# [,1]
# [1,] 0.8064807807

#The correlation has increased to 0.80 which is bettern than the previous result. We can improve this result by applying different number of
#hidden nodes, using different activation function etc...

####################################### Functions ################################################
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}
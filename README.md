# Coursera---Practical-Machine-Learning
Coursera - Practical Machine Learning

## Introduction
The goal of this project is to predict the manner in which the participants did the exercise using the available data.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The 6 different classes - 
Class A - Exactly according to the specification
Class B - Throwing the elbows to the front
Class C - Lifting the dumbbell only halfway
Class D - Lowering the dumbbell only halfway
Class E - Throwing the hips to the front


## Getting and cleaning the data
Download the data, load it into R and prepare it for the modeling process.

```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(e1071)
library(randomForest)
set.seed(1)

setwd("C:/Users/deepak.d.arya/Desktop/Trainings/Data Science - Coursera/Practical Machine learning")
train.url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url <- 
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

```

## Read the files into memory 
```{r}
train.data.raw <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test.data.raw <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

## Remove unnecessary columns
```{r}
# Drop the first 7 columns as they're unnecessary for predicting.
train.data.clean1 <- train.data.raw[,8:length(colnames(train.data.raw))]
test.data.clean1 <- test.data.raw[,8:length(colnames(test.data.raw))]

# Drop colums with NAs
train.data.clean1 <- train.data.clean1[, colSums(is.na(train.data.clean1)) == 0] 
test.data.clean1 <- test.data.clean1[, colSums(is.na(test.data.clean1)) == 0] 

# Check for near zero variance predictors and drop them if necessary
nzv <- nearZeroVar(train.data.clean1,saveMetrics=TRUE)
zero.var.ind <- sum(nzv$nzv)

if ((zero.var.ind>0)) {
        train.data.clean1 <- train.data.clean1[,nzv$nzv==FALSE]
}
```

## Slicing the data for cross validation
```{r}
in.training <- createDataPartition(train.data.clean1$classe, p=0.70, list=F)
train.data.final <- train.data.clean1[in.training, ]
validate.data.final <- train.data.clean1[-in.training, ]
```

## Model development

### Train the model

The training data-set is used to fit a Random Forest model because it automatically selects important variables and is robust to correlated covariates & outliers in general. 5-fold cross validation is used when applying the algorithm. A Random Forest algorithm is a way of averaging multiple deep decision trees, trained on different parts of the same data-set, with the goal of reducing the variance. This typically produces better performance at the expense of bias and interpret-ability. The Cross-validation technique assesses how the results of a statistical analysis will generalize to an independent data set. In 5-fold cross-validation, the original sample is randomly partitioned into 5 equal sized sub-samples. a single sample is retained for validation and the other sub-samples are used as training data. The process is repeated 5 times and the results from the folds are averaged.
```{r}
control.parms <- trainControl(method="cv", 5)
rf.model <- train(classe ~ ., data=train.data.final, method="rf",
                 trControl=control.parms, ntree=251)
rf.model
```

### Estimate Performance

The model fit using the training data is tested against the validation data. Predicted values for the validation data are then compared to the actual values. This allows forecasting the accuracy and overall out-of-sample error, which indicate how well the model will perform with other data.

```{r}
rf.predict <- predict(rf.model, validate.data.final)
confusionMatrix(validate.data.final$classe, rf.predict)
```

```{r}
accuracy <- postResample(rf.predict, validate.data.final$classe)
acc.out <- accuracy[1]

overall.ose <- 
        1 - as.numeric(confusionMatrix(validate.data.final$classe, rf.predict)
                       $overall[1])
```

### Results
The accuracy of this model is 0.9940527 and the Overall Out-of-Sample error is 0.0059473.

## Run the model
The model is applied to the test data to produce the results.

```{r}
results <- predict(rf.model, 
                   test.data.clean1[, -length(names(test.data.clean1))])
results
```

## Appendix - Decision Tree

```{r}
treeModel <- rpart(classe ~ ., data=train.data.final, method="class")
fancyRpartPlot(treeModel)
```

    library(ggplot2)
    library(dplyr)
    library(caret)
    library(randomForest)

    trainraw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", sep=",",header = TRUE,na.strings = c("NA","",'#DIV/0!'))
    testraw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", sep=",",header = TRUE, na.strings = c("NA","",'#DIV/0!'))

R Markdown
----------

First, we dismissed all variables with no information or just NA as
content for test and train datasets.

    train<- trainraw[,(colSums(is.na(trainraw)) == 0)]
    dim(train)

    ## [1] 19622    60

    test<- testraw[,(colSums(is.na(testraw)) == 0)]
    dim(test)

    ## [1] 20 60

We see that 100 columns were dropped for each dataset from raw data, we
had 160 variables, and after this first cleansing, we kept only 60
covariates.

After that, we eliminate the covariates with near zero variance usig
`caret` package from train and test datasets because they do not add so
much value to our predictions. Only 54 covariates remain in the
datasets.

    set.seed(999)
    no_var <- nearZeroVar(train,saveMetrics=TRUE)
    traincl <- train[,no_var$nzv==FALSE]
    testcl <- test[,no_var$nzv==FALSE]

    dim(traincl)

    ## [1] 19622    59

    dim(testcl)

    ## [1] 20 59

To execute the cross validation, first we divided the training dataset
75% for a new training set and the rest for a validation set.

    train_idx <- createDataPartition(y=traincl$classe, p=0.75, list=F)
    train_data <- traincl[train_idx, ]
    valid_data <- traincl[-train_idx, ]

    train_data <- train_data[, -(1:5)]
    valid_data <- valid_data[, -(1:5)]
    test_data <- testcl[, -(1:5)]

Now, using Cross Validation we fitted our model using a random forest
decision tree with 3 folds.

    fitControl <- trainControl(method="cv", number=3, verboseIter=F)

    fit <- train(classe ~ ., data=train_data, method="rf", trControl=fitControl, importance=TRUE)

    fit

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9812, 9812, 9812 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9917788  0.9895996
    ##   27    0.9960592  0.9950154
    ##   53    0.9927979  0.9908895
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

For interpretation purposes, we plotted the importance of each
individual variable and as we can see, the variable `roll_belt` has the
greatest importance in this model.

    varImpPlot(fit$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, cex = 0.6, main = "Importance of the Individual Principal Components")

![](CourseProject_files/figure-markdown_strict/unnamed-chunk-7-1.png)

Once we had our fit, we built our confusion matrix with our predictions
from the random forest model and the validation dataset.

    predictRf <- predict(fit, valid_data)
    confusionMatrix(valid_data$classe, predictRf)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    0    0    0    1
    ##          B    3  944    2    0    0
    ##          C    0    1  854    0    0
    ##          D    0    0    2  802    0
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.998          
    ##                  95% CI : (0.9963, 0.999)
    ##     No Information Rate : 0.2849         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9974         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9979   0.9989   0.9953   0.9988   0.9989
    ## Specificity            0.9997   0.9987   0.9998   0.9995   0.9998
    ## Pos Pred Value         0.9993   0.9947   0.9988   0.9975   0.9989
    ## Neg Pred Value         0.9991   0.9997   0.9990   0.9998   0.9998
    ## Prevalence             0.2849   0.1927   0.1750   0.1637   0.1837
    ## Detection Rate         0.2843   0.1925   0.1741   0.1635   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9988   0.9988   0.9975   0.9991   0.9993

Also, we computed the accuracy of the model, that as we see is almost
99.80%.

    accuracy <- postResample(predictRf, valid_data$classe)
    accuracy

    ##  Accuracy     Kappa 
    ## 0.9979608 0.9974207

Conversely, the out-of-sample error is quite small, just 0.20%.

    oose <- 1 - as.numeric(confusionMatrix(valid_data$classe, predictRf)$overall[1])
    oose

    ## [1] 0.002039152

Finally, we predicted the 20 test cases available in the test data.

    result <- predict(fit, test_data[, -length(names(test_data))])
    result

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

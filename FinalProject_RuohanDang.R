## Library
library(tidyverse)
library(ggplot2)
library(glmnet)
library(caret)
library(e1071)
library(party)
library(class)
library(randomForest)
library(neuralnet)

RNGkind(sample.kind = "Rounding")
##########################################
## Import dataset and check ##
##########################################
# training data
train_data <- read.csv("train.csv")
head(train_data,5)
summary(train_data)
dim(train_data)
# test data
test_data <- read.csv("test.csv")
head(test_data,5)
summary(test_data)
dim(test_data)
# store the response variable
activity_train <- train_data$Activity
activity_test <- test_data$Activity

# combine two data sets for EDA
overall_data <- rbind(train_data,test_data)
glimpse(train_data)
dim(overall_data)

##########################################
## Data Preprocessing for EDA ##
##########################################
# Checking for missing values
sum(is.na(train_data))
sum(is.na(test_data))
# there are no missing values

# Checking for duplicates
summary(duplicated(train_data))
summary(duplicated(test_data))

# Checking for class imbalance
table(overall_data$Activity)
ggplot(data=overall_data, aes(x=Activity, color=Activity, fill=Activity)) + 
  geom_bar() + labs(x= "Activities",title = "Smartphone activity labels distribution")
# There is almost same number of observations across all the six activities 
# so this data does not have class imbalance problem.

# frequency of 'Subject' Variable
table(overall_data$subject)
ggplot(overall_data, aes(subject)) + 
  geom_bar(fill="gray") + ylim(0,500) +
  labs(title = "The distribution of 30 subjects in the overall dataset")

# Static and dynamic activities
ggplot(overall_data, aes(x=tBodyAccMag.mean.., color=Activity)) + geom_density() +
  labs(title = "Static and dynamic activities by tBodyAccMag-mean density plot")

# figure out the variation in data for each activity
# Mean Body Acceleration
subject20 <- overall_data[overall_data$subject==20,]
ggplot(subject20, aes(x=Activity, y=tBodyAcc.mean...X, color=Activity)) + 
  geom_point() + labs(title = "Subject Mean Body Acceleration - X axis scatter plot")
ggplot(subject20, aes(x=Activity, y=tBodyAcc.mean...Y, color=Activity)) + 
  geom_point() + labs(title = "Subject Mean Body Acceleration - Y axis scatter plot")
ggplot(subject20, aes(x=Activity, y=tBodyAcc.mean...Z, color=Activity)) + 
  geom_point() + labs(title = "Subject Mean Body Acceleration - Z axis scatter plot")

# Maximum Body Acceleration
ggplot(subject20, aes(x=Activity, y=tBodyAcc.max...X, color=Activity)) + 
  geom_point() + labs(title = "Subject Max Body Acceleration - X axis scatter plot")
ggplot(subject20, aes(x=Activity, y=tBodyAcc.max...Y, color=Activity)) + 
  geom_point() + labs(title = "Subject Max Body Acceleration - Y axis scatter plot")
ggplot(subject20, aes(x=Activity, y=tBodyAcc.max...Z, color=Activity)) + 
  geom_point() + labs(title = "Subject Max Body Acceleration - Z axis scatter plot")

# Mean Gravity Acceleration signals
ggplot(subject20, aes(x=Activity, y=tGravityAcc.mean...X, color=Activity)) + 
  geom_point() + labs(y="tGravityAcc.mean.X",title = "Angle (X, GravityMean) Scatter Plot")
ggplot(subject20, aes(x=Activity, y=tGravityAcc.mean...Y, color=Activity)) + 
  geom_point() + labs(y="tGravityAcc.mean.Y",title = "Angle (Y, GravityMean) Scatter Plot")
ggplot(subject20, aes(x=Activity, y=tGravityAcc.mean...Z, color=Activity)) + 
  geom_point() + labs(y="tGravityAcc.mean.Y",title = "Angle (Z, GravityMean) Scatter Plot")

##########################################
## Drop some feature ###
##########################################
# The variable "subject" is just a label, so we can romove it for feature selection
train_data <- train_data %>% 
  select(-"subject")
test_data <- test_data %>% 
  select(-"subject")

##########################################
## Modeling ###
##########################################
# GLM with lasso, using cross validation
# create a calibration and training set from the original training set
set.seed(327) #set seed
index <- sample(x=nrow(train_data), size=.80*nrow(train_data))
train_sub <- train_data[index, ]
test_sub <- train_data[-index, ]

dim(train_sub)
dim(test_sub)

train_mat <- model.matrix(Activity~., train_sub)[,-1]
lasso.fit <- glmnet(train_mat, train_sub$Activity, family = "multinomial", type.multinomial = "grouped")
plot(lasso.fit, xvar = "lambda", label = TRUE, type.coef = "2norm")
plot(lasso.fit, xvar = "dev", label = TRUE)

# use cross-validation to find estimate the best lambda with the lowest test-error, used 5-fold
cvfit <- cv.glmnet(train_mat, train_sub$Activity, family="multinomial", type.measure="class", 
                   type.multinomial = "grouped", nfolds = 5, parallel = TRUE)
plot(cvfit)
bestlam <- cvfit$lambda.min
log(bestlam)

test_mat <- model.matrix(Activity~., test_sub)[,-1]
lasso.pred <- predict(cvfit, s = bestlam, newx = test_mat, type = "class")
table(lasso.pred, test_sub$Activity)
acc <- sum(lasso.pred == test_sub$Activity) / length(lasso.pred)
acc

# we can perform on original test set
test.pred <- predict(cvfit, s = bestlam, newx = model.matrix(Activity~., test_data)[,-1], type = "class")
test.pred <- as.factor(test.pred)
p_lasso <- mean(test.pred == test_data$Activity)
lassoconfusionMatrix(test.pred, test_data$Activity)

# Support vector machines
svm_linear <- svm(Activity~ ., data=train_data, type='C-classification', kernel='linear')

# Naive Bayes
bnc <- naiveBayes(Activity~ ., data=train_data)

# Decision Trees
ctree <- ctree(Activity ~ ., data=train_data)

# KNN
knn9 <- knn(train = train_data[,1:561], test = test_data[,1:561], cl = train_data$Activity, k=9)

# Random Forest
rf <- randomForest(Activity ~ ., data = train_data)


### Test set predictions
# Support vector machines
pred_test_svm <- predict(svm_linear, test_data)
p_svm <- mean(pred_test_svm == test_data$Activity)

# Naive Bayes
pred_test_bnc <- predict(bnc, test_data)
p_bnc <- mean(pred_test_bnc == test_data$Activity)

# Decision Trees
pred_test_ctree <- predict(ctree, test_data)
p_ctree <- mean(pred_test_ctree == test_data$Activity)

# KNN
p_knn9 <- mean(knn9 == test_data$Activity)

# Random Forest
pred_test_rf <- predict(rf, test_data)
p_rf <- mean(pred_test_rf == test_data$Activity)


# Confusion Matrix Analysis
caret::confusionMatrix(test.pred, test_data$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_svm, test_data$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_bnc, test_data$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_ctree, test_data$Activity, positive="1", mode="everything")
caret::confusionMatrix(knn9, test_data$Activity, positive="1", mode="everything")
caret::confusionMatrix(pred_test_rf, test_data$Activity, positive="1", mode="everything")


####################
##### Conclusion ###
####################
# plot  
p <- as.vector(c(p_lasso, p_svm, p_bnc, p_ctree, p_knn9, p_rf))
p_round <- round(p,4)
barplot(p_round, ylim = c(0,1), main = "Model comparison", 
        xlab = "Models", names.arg = c("GLM/Lasso", "SVM","Naive Bayes","Decision Trees","KNN","Randomforest"),
        col = c("darkseagreen", "darkslategray3", "gold2", "lightpink2", "lightsalmon1", "mediumseagreen"))
text(p, labels = as.character(p_round), pos = 3, cex = 0.75)

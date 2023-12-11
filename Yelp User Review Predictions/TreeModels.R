#Clear
cat("\014")  
rm(list=ls())

library (glmnet) 
library(caret)
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(tree)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest) 

# Data on User Reviews (review_data_small)
load("~/yelp_review_small.Rda")

# Data on the Users (user_data_small)
load("~/yelp_user_small.Rda")

# Data on the Businesses (business_data)
business_data <- stream_in(file("~/yelp_academic_dataset_business.json")) 

# Merge data sets
data_initial_merge <- merge(review_data_small, user_data_small, by = "user_id")
data_merged <- merge(data_initial_merge, business_data, by = "business_id")

data_final <- data_merged %>% select(stars.x, funny.x, cool.x, review_count.x, useful.y, 
                                     funny.y, cool.y, fans, average_stars, stars.y, review_count.y, 
                                     is_open, starts_with('compliment_'))

# Make categorical:
data_final$stars.x <- as.factor(data_final$stars.x)

# Split data
set.seed(1)
test_set_size <- 10000
test <- sample(1:nrow(data_final), test_set_size) 
data_final_test <- data_final[test, ]
data_final_train <- data_final[-test, ]

# Actual class labels
true_values <- data_final_test$stars.x

# Random Forest 
model_RF<-randomForest(stars.x ~ ., data=data_final_train, ntree=50, importance = TRUE, nodesize = 5)
model_RF2<-randomForest(stars.x ~ ., data=data_final_train, ntree=50, importance = TRUE, nodesize = 1)

model_RF
model_RF2

plot(model_RF)
varImpPlot(model_RF)

# Make RF predictions
predictions_rf <- predict(model_RF, data_final_test)

# Calculate RF accuracy
accuracy_rf <- sum(predictions_rf == true_values) / length(true_values)
print(paste("RF Accuracy:", accuracy_rf))

# Regression Tree with rpart:
rpart_tree <- rpart(stars.x ~ ., data = data_final_train, method = "class")
rpart.plot(rpart_tree)

# Make rpart predictions
predictions_rpart <- predict(rpart_tree, data_final_test, type = "class")

# Calculate rpart accuracy
accuracy_rpart <- sum(predictions_rpart == true_values) / length(true_values)
print(paste("Accuracy:", accuracy_rpart))
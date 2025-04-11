############################################################################################
############################################################################################
# Install all the packages
#install.packages('mlr')
#install.packages('naniar')
#install.packages('corrplot')
#install.packages('neuralnet')
#install.packages('NeuralNetTools')
#install.packages('ROCR')
#install.packages('ModelMetrics')
#install.packages('caret')
#install.packages('doMC')
#install.packages('doParallel')
#install.packages('randomForest')
#install.packages('DMwR')
#install.packages('snow')
#install.packages('xgboost')
#install.packages('microbenchmark')
#install.packages('smbinning')
#install.packages('glmnet')
# Install lightgbm manually: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#windows
############################################################################################
############################################################################################
## This code expects the following pre-defined names
# Set path that contains the functions
macros_path = "C:/Data_Science/R_Machine_Learning_Suite/src/functions" 
# Set the path that contains the tables relevant to this code
data_path = "C:/Data_Science/R_Machine_Learning_Suite/data"
# Set the target variable name in the original dataset
target_variable_name = 'bad_flag'
# Set the weight variable name in the original dataset
weight_variable_name = 'weight'
# Set the (fraud) amount variable name in the original dataset
amount_variable_name = 'amount'
# Set the ID variable name in the original dataset
ID_variable_name = 'transact_id'
############################################################################################
############################################################################################
# Load data using the 'fread' function for better performance
input_data_path = paste(data_path, "/input/Original_table.csv", sep="")
library(data.table)
input.table = fread(input_data_path)

head(input.table)
str(input.table)

# Replace missing values with NA because the fread function in R does not do this properly, and also convert to character
library(naniar)
library(dplyr)
vars_to_change_to_character = c("character1", "character2", "character3", "character4", "character5", "character6", "character7", "character8", target_variable_name)

input.table.tmp = 
  input.table %>%
       replace_with_na(replace=list(
         character1=c('', 'NA')
         , character2=c('', 'NA')
         , character3=c('', 'NA')
         , character4=c('', 'NA')
         , character5=c('', 'NA')
         , character6=c('', 'NA')
         , character7=c('', 'NA')
         , character8=c('', 'NA')
       )) %>%
      mutate_each(funs(as.character), vars_to_change_to_character)
input.table = as.data.frame(input.table.tmp)
str(input.table)
############################################################################################
############################################################################################
# Perform Exploratory analysis
str(input.table)
summary(input.table)

# Check missingness
sort(apply(input.table, 2, function(x) sum(is.na(x))) / nrow(input.table), decreasing=T)

# Check if the dataset is unbalanced
library(ggplot2)
common_theme = theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p = ggplot(input.table, aes(x = bad_flag)) + geom_bar() + ggtitle("Number of class labels") + common_theme
print(p)

# Remove variables that have high missing percentage
#character4
#numeric3
# Remove variables that are not predictors or target
#weight_variable_name 
#ID_variable_name
# Remove categorical variables with too many levels
#character3
#character7
#character8
# Remove categorical variables with only one level or standard deviation 0
#character5
#numeric5
drop_variables = c("character4", "numeric3", 
                   weight_variable_name, ID_variable_name, 
                   "character3", "character7", "character8", 
                   "character5", "numeric5")
input.table = input.table[, !names(input.table) %in% drop_variables]

# Check missingness
sort(apply(input.table, 2, function(x) sum(is.na(x))) / nrow(input.table), decreasing=T)

# Replace missing values from numeric variables with mean
numeric_vars = names(select_if(input.table, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]
for(i in numeric_vars){
# Find the location of the variable in the data frame  
  loc = which(names(input.table)==i)
  input.table[is.na(input.table[,loc]), loc] = mean(input.table[,loc], na.rm = TRUE)
}

# Replace missing values from character variables with 'M'
character_vars = names(select_if(input.table, is.character))
character_vars = character_vars[character_vars != target_variable_name]
for(i in character_vars){
# Find the location of the variable in the data frame  
  loc = which(names(input.table)==i)
  input.table[is.na(input.table[,loc]), loc] = 'M'
}

# Check missingness
sort(apply(input.table, 2, function(x) sum(is.na(x))) / nrow(input.table), decreasing=T)
############################################################################################
############################################################################################
# Data Partition
set.seed(222)
# Convert character variables to factor variables - this is needed for some functions to operate, e.g. createDummyFeatures
input.table.tmp = 
  input.table %>%
  mutate_each(funs(as.factor), c(character_vars, target_variable_name))
input.table = as.data.frame(input.table.tmp)

ind = sample(2, nrow(input.table), replace = TRUE, prob = c(0.7, 0.3))
input.table.training = input.table[ind==1,]
input.table.testing = input.table[ind==2,]
############################################################################################
############################################################################################
# Produce Information Value table
library(smbinning)
iv_df = data.frame(VARS=c(numeric_vars, character_vars), IV=rep(0, length(c(numeric_vars, character_vars))))  # init for IV results

smb.df = input.table.training[, names(input.table.training) %in% c(numeric_vars, character_vars, target_variable_name)]
smb.df[, names(smb.df)==target_variable_name] = as.numeric(smb.df[, names(smb.df)==target_variable_name])
smb.df[, names(smb.df)==target_variable_name] = smb.df[, names(smb.df)==target_variable_name] - 1

# Calcualte IV for the numeric variables only
for(num_var in numeric_vars){
  smb = smbinning(smb.df, y=target_variable_name, x=num_var, p=0.05)
  if(class(smb) != "character"){  # any error while calculating scores.
    iv_df[iv_df$VARS == num_var, "IV"] <- smb$iv
  }else{
    iv_df[iv_df$VARS == num_var, "IV"] <- smb
  }
}

# Calcualte IV for the character variables only
for(fac_var in character_vars){
  smb = smbinning.factor(smb.df, y=target_variable_name, x=fac_var, maxcat=50)
  if(class(smb) != "character"){  # any error while calculating scores.
    iv_df[iv_df$VARS == fac_var, "IV"] <- smb$iv
  }else{
    iv_df[iv_df$VARS == fac_var, "IV"] <- smb
  }
}

# Print IV table
iv_df = iv_df[order(iv_df$IV, decreasing=T),]

# Select variables based on IV threshold cutoff
iv_threshold = 0.001
iv_vars = suppressWarnings(as.character(iv_df[!is.na(iv_df[as.numeric(iv_df$IV)>iv_threshold & as.numeric(iv_df$IV)<10^10,])[,1],]$VARS))
input.table.training = input.table.training[,names(input.table.training) %in% c(iv_vars, target_variable_name)]
input.table.testing = input.table.testing[,names(input.table.testing) %in% c(iv_vars, target_variable_name)]

# Select numeric variables
numeric_vars = names(select_if(input.table.training, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]

# Select character variables
character_vars = names(select_if(input.table.training, is.factor))
character_vars = character_vars[character_vars != target_variable_name]
############################################################################################
############################################################################################
# Create dummy variables
library(mlr)
input.table.dummy.training = createDummyFeatures(input.table.training[, names(input.table.training) %in% character_vars], method='reference')
input.table.dummy.training$target_variable_name = input.table.training[, which(names(input.table.training)==target_variable_name)]
names(input.table.dummy.training)[which(names(input.table.dummy.training)=="target_variable_name")] = target_variable_name

input.table.dummy.testing = createDummyFeatures(input.table.testing[, names(input.table.testing) %in% character_vars], method='reference')
input.table.dummy.testing$target_variable_name = input.table.testing[, which(names(input.table.testing)==target_variable_name)]
names(input.table.dummy.testing)[which(names(input.table.dummy.testing)=="target_variable_name")] = target_variable_name

# Correlation analysis
library(corrplot)
corr_plot = corrplot(cor(select_if(input.table.dummy.training, is.numeric)), method = "circle", type = "upper")

# Normalize data to interval {-1, 1}
numeric_vars = names(select_if(input.table.training, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]
input.table.training.numeric = input.table.training[, which(names(input.table.training) %in% numeric_vars)]
minValue.training = sapply(input.table.training.numeric, min)
maxValue.training = sapply(input.table.training.numeric, max)
normalized.data.training = as.data.frame(scale(input.table.training.numeric, 
      center = (maxValue.training+minValue.training)/2, 
      scale = (maxValue.training-minValue.training)/2))
input.table.dummy.normalized.training = cbind(input.table.dummy.training, normalized.data.training)

numeric_vars = names(select_if(input.table.testing, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]
input.table.testing.numeric = input.table.testing[, which(names(input.table.testing) %in% numeric_vars)]
minValue.testing = sapply(input.table.testing.numeric, min)
maxValue.testing = sapply(input.table.testing.numeric, max)
normalized.data.testing = as.data.frame(scale(input.table.testing.numeric, 
                                               center = (maxValue.testing+minValue.testing)/2, 
                                               scale = (maxValue.testing-minValue.testing)/2))
input.table.dummy.normalized.testing = cbind(input.table.dummy.testing, normalized.data.testing)

# Correlation analysis
library(corrplot)
corr_plot = corrplot(cor(select_if(input.table.dummy.normalized.training, is.numeric)), method = "circle", type = "upper")
cor(input.table.dummy.normalized.training[,51:53])
# numeric6 and numeric8 are perfectely correlated, so remove one of them
input.table.dummy.normalized.training = input.table.dummy.normalized.training[, !names(input.table.dummy.normalized.training) %in% c("numeric8","character6.9")]
input.table.dummy.normalized.testing = input.table.dummy.normalized.testing[, !names(input.table.dummy.normalized.testing) %in% c("numeric8","character6.9")]

# Standardize data
numeric_vars = names(select_if(input.table.training, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]
standardized.data.training = as.data.frame(scale(input.table.training[, which(names(input.table.training) %in% numeric_vars)]))
input.table.dummy.standardized.training = cbind(input.table.dummy.training, standardized.data.training)

numeric_vars = names(select_if(input.table.testing, is.numeric))
numeric_vars = numeric_vars[numeric_vars != target_variable_name]
standardized.data.testing = as.data.frame(scale(input.table.testing[, which(names(input.table.testing) %in% numeric_vars)]))
input.table.dummy.standardized.testing = cbind(input.table.dummy.testing, standardized.data.testing)

# Correlation analysis
library(corrplot)
corr_plot = corrplot(cor(select_if(input.table.dummy.standardized.training, is.numeric)), method = "circle", type = "upper")
cor(input.table.dummy.standardized.training[,51:53])
# numeric6 and numeric8 are perfectely correlated, so remove one of them
input.table.dummy.standardized.training = input.table.dummy.standardized.training[, !names(input.table.dummy.standardized.training) %in% c("numeric8","character6.9")]
input.table.dummy.standardized.testing = input.table.dummy.standardized.testing[, !names(input.table.dummy.standardized.testing) %in% c("numeric8","character6.9")]
############################################################################################
############################################################################################
# Set formula
predictor.names = paste(names(input.table.dummy.normalized.training[,  !names(input.table.dummy.normalized.training) %in% target_variable_name]), collapse = "+")
formula = as.formula(paste(target_variable_name, predictor.names, sep = "~"))

# VIF analysis - remove variables with VIF>10
library(car)
lm.fit <- glm(formula, data = input.table.dummy.normalized.training, family=binomial(link='logit'))
sort(vif(lm.fit), decreasing=T)

# Variable reduction using stepwise regression
#library(MASS)
#lm.fit.AIC = stepAIC(lm.fit, direction='both')
#formula.AIC = formula(lm.fit.AIC)

# Variable reduction using LASSO regression
library(glmnet)
lasso.fit = glmnet(as.matrix(input.table.dummy.normalized.training[,  !names(input.table.dummy.normalized.training) %in% target_variable_name]), as.vector(input.table.dummy.normalized.training[,  names(input.table.dummy.normalized.training) %in% target_variable_name]), family = "binomial", alpha=1)
# Plot variable coefficients vs. shrinkage parameter lambda
plot(lasso.fit, xvar='lambda')
plot(log(lasso.fit$lambda), lasso.fit$dev.ratio)

# Choosing the lambda value
cv.lasso.fit = cv.glmnet(as.matrix(input.table.dummy.normalized.training[,  !names(input.table.dummy.normalized.training) %in% target_variable_name]), as.vector(input.table.dummy.normalized.training[,  names(input.table.dummy.normalized.training) %in% target_variable_name]), family = "binomial", alpha=1, nfolds=10)
plot(cv.lasso.fit)
c = coef(cv.lasso.fit, s = "lambda.1se")[,1][coef(cv.lasso.fit, s = "lambda.1se")[,1]!=0]
best.coef = names(c)[!names(c) %in% "(Intercept)"]
formula.best.lasso = as.formula(paste(target_variable_name, paste(best.coef, collapse = "+"), sep = "~"))
############################################################################################
############################################################################################
library(h2o)
# Use all the CPUs
localH2O = h2o.init(nthreads = -1)

h2o.init()

# Load the data in H2O
input.table.training.h2o = as.h2o(input.table.dummy.normalized.training)
input.table.testing.h2o = as.h2o(input.table.dummy.normalized.testing)

#dependent variable (Purchase)
y.dep = 49
#independent variables (dropping ID variables)
x.indep = c(1:48,50:54)
############################################################################################
############################################################################################

############################################################################################
############################################################################################
# Load the model evaluation functions
evaluation_functions = paste(macros_path, "/evaluation.R", sep="")
source(evaluation_functions)
############################################################################################
############################################################################################
# Random Forest with H2O
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, training_frame=input.table.training.h2o, 
                                    ntrees = 100, # ntrees: 20 - 400 (number of trees)
                                    max_depth = 5, # max_depth: 2 - 20 (depth of the tree, 5-6 is usually default)
                                    min_rows = 50, # min_rows: 10 - 500 (minimum number of observations in each leaf, default 10)
                                    nbins = 30, # nbins: 10, 20, 30 (number of bins to split, for numerical variables only)
                                    nbins_cats = 20, # nbins_cats: 5, 10, 15, 20 (number of bins to split, for categorical variables only)
                                    sample_rate = 0.7, # sample_rate: 0 - 1 (row sample rate, default 0.63)
                                    col_sample_rate_per_tree = 0.7, # col_sample_rate_per_tree: 0.0 - 1.0  (Specify the column sample rate per tree, defaults to 1)
                                    nfolds = 0, # nfolds: number of folds for cross-validation, set 0 for no cross-validation, set >1 if cross-validation is desired
                                    seed = 1122)
)
#user  system elapsed 
#0.13    0.01    2.39 

# Check performance
h2o.performance(rforest.model)

# Check variable importance
h2o.varimp(rforest.model)
h2o.varimp_plot(rforest.model)
############################################################################################
# Performance statistics on training sample
library(ROCR)
p1.training.output.rf.h2o = as.data.frame(h2o.predict(rforest.model, newdata=input.table.training.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.rf.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.rf.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.rf.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.rf.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.rf.h2o,
                    threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
p1.testing.output.rf.h2o = as.data.frame(h2o.predict(rforest.model, newdata=input.table.testing.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.rf.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.rf.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.rf.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.rf.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.rf.h2o,
                                     threshold=0.5)
############################################################################################
############################################################################################

############################################################################################
############################################################################################# 
# GBM with H2O
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = input.table.training.h2o, 
                       ntrees = 100, # ntrees: 20 - 400 (number of trees)
                       max_depth = 2, # max_depth: 2 - 20 (depth of the tree, 5-6 is usually default)
                       min_rows = 20, # min_rows: 10 - 500 (minimum number of observations in each leaf, default 10)
                       nbins = 30, # nbins: 10, 20, 30 (number of bins to split, for numerical variables only)
                       nbins_cats = 20, # nbins_cats: 5, 10, 15, 20 (number of bins to split, for categorical variables only)
                       learn_rate = 0.1, # learn_rate: 0.05, 0.10, 0.20, 0.30 (learing rate. Shrinkage is used for reducing, or shrinking, the impact of each additional fitted base-learner (tree). It reduces the size of incremental steps and thus penalizes the importance of each consecutive iteration. The intuition behind this technique is that it is better to improve a model by taking many small steps than by taking fewer large steps. If one of the boosting iterations turns out to be erroneous, its negative impact can be easily corrected in subsequent steps. High learn rates and especially values close to 1.0 typically result in overfit models with poor performance.  Values much smaller than .01 significantly slow down the learning process and might be reserved for overnight runs. 0.10 default)
                       sample_rate = 0.7, # sample_rate: 0 - 1 (row sample rate)
                       col_sample_rate = 0.7, # col_sample_rate: 0 - 1 (column sample rate per split)
                       nfolds = 0, # nfolds: number of folds for cross-validation, set 0 for no cross-validation, set >1 if cross-validation is desired
                       seed = 1122)
)
#user  system elapsed 
#0.14    0.08    2.38 

# Check performance
h2o.performance(gbm.model)

# Check variable importance
h2o.varimp(gbm.model)
h2o.varimp_plot(gbm.model)
############################################################################################
# Performance statistics on training sample
library(ROCR)
p1.training.output.gbm.h2o = as.data.frame(h2o.predict(gbm.model, newdata=input.table.training.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.gbm.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.gbm.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.gbm.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                p1_prediction_df=p1.training.output.gbm.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                                     p1_prediction_df=p1.training.output.gbm.h2o,
                                     threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
p1.testing.output.gbm.h2o = as.data.frame(h2o.predict(gbm.model, newdata=input.table.testing.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.gbm.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.gbm.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.gbm.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.gbm.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.gbm.h2o,
                                     threshold=0.5)
############################################################################################
############################################################################################

############################################################################################
############################################################################################
# Deep Learning with H2O
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = input.table.training.h2o,
                                      epoch = 60,
                                      hidden = c(3,2),
                                      activation = "Rectifier",
                                      nfolds = 0, # nfolds: number of folds for cross-validation, set 0 for no cross-validation, set >1 if cross-validation is desired
                                      seed = 1122
  )
)
#user  system elapsed 
#0.24    0.01    1.38 

# Check performance
h2o.performance(dlearning.model)

# Check variable importance
h2o.varimp(dlearning.model)
h2o.varimp_plot(dlearning.model)
############################################################################################
# Performance statistics on training sample
library(ROCR)
p1.training.output.dl.h2o = as.data.frame(h2o.predict(dlearning.model, newdata=input.table.training.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.dl.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.dl.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.dl.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                p1_prediction_df=p1.training.output.dl.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                                     p1_prediction_df=p1.training.output.dl.h2o,
                                     threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
p1.testing.output.dl.h2o = as.data.frame(h2o.predict(dlearning.model, newdata=input.table.testing.h2o))[,3]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.dl.h2o)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.dl.h2o)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.dl.h2o)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.dl.h2o)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.dl.h2o,
                                     threshold=0.5)
############################################################################################
############################################################################################



############################################################################################
############################################################################################
############################################################################################
############################################################################################
# SMOTE cross-validation (Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall and W. Philip Kegelmeyer's "SMOTE: Synthetic Minority Over-sampling Technique" (Journal of Artificial Intelligence Research, 2002, Vol. 16, pp. 321-357, 
# https://rpubs.com/slazien/fraud_detection) and also Random Forest and GBM

# Set random seed for reproducibility
set.seed(42)
# Parallel processing for faster training in Linux
#library(doMC)
#registerDoMC(cores = 8)
# Parallel processing for faster training in Windows
library(doParallel)
#the following line will create a local 4-node snow cluster
workers=makeCluster(4,type="SOCK")
registerDoParallel(workers)
foreach(i=1:4) %dopar% Sys.getpid()

# Use 10-fold cross-validation and SMOTE
library(caret)
system.time(
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",
                     summaryFunction = twoClassSummary,
                     savePredictions = T)
)
############################################################################################
############################################################################################
# Train a Random Forest classifier, maximising recall (sensitivity)
# Change the target variable for the training dataset to run the RF algorithm
levels(input.table.dummy.normalized.training[, which(names(input.table.dummy.normalized.training)==target_variable_name)]) = make.names(c(0, 1))

library(randomForest)
system.time(
model_rf_smote <- train(formula, data = input.table.dummy.normalized.training, method = "rf", 
                        
                       trControl = ctrl, verbose = T, metric = "ROC")
)
#user  system elapsed 
#58.86    3.47  777.06 

# Convert back the target variable for the training dataset as it was before the conversion
input.table.dummy.normalized.training[, which(names(input.table.dummy.normalized.training)==target_variable_name)] = as.factor(ifelse(input.table.dummy.normalized.training[, which(names(input.table.dummy.normalized.training)==target_variable_name)]=='X0', '0', '1'))
############################################################################################
# Model performance specific statistics on training data
library(ROCR)
p1.training.output.rf.smote = predict(model_rf_smote, input.table.dummy.normalized.training, type = "prob")[,2]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.rf.smote)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.rf.smote)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.rf.smote)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                p1_prediction_df=p1.training.output.rf.smote)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                                     p1_prediction_df=p1.training.output.rf.smote,
                                     threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
p1.testing.output.rf.smote = predict(model_rf_smote, input.table.dummy.normalized.testing, type = "prob")[,2]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.rf.smote)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.rf.smote)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.rf.smote)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.rf.smote)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.rf.smote,
                                     threshold=0.5)
############################################################################################
############################################################################################

############################################################################################
############################################################################################
# Train data using XGBoost
library(xgboost)
y.train = as.numeric(input.table.dummy.normalized.training[, names(input.table.dummy.normalized.training)==target_variable_name])
y.train = ifelse(y.train==1, 0, 1)
y.test = as.numeric(input.table.dummy.normalized.testing[, names(input.table.dummy.normalized.testing)==target_variable_name])
y.test = ifelse(y.test==1, 0, 1)

xgboost.training = xgb.DMatrix(data = as.matrix(input.table.dummy.normalized.training[, !names(input.table.dummy.normalized.training)==target_variable_name]), label = y.train)
xgboost.testing = xgb.DMatrix(data = as.matrix(input.table.dummy.normalized.testing[, !names(input.table.dummy.normalized.testing)==target_variable_name]), label = y.test)

xgb = xgboost(data = xgboost.training, 
              nrounds = 100, 
              gamma = 0.1, 
              max_depth = 10, 
              objective = "binary:logistic", 
              nthread = 7, 
              tree_method='hist')
library(microbenchmark)
system.time(
xgb.bench.hist <- microbenchmark(
  xgb.model.hist <- xgb.train(data = xgboost.training
                              , params = list(objective = "binary:logistic"
                                              , eta = 0.1
                                              , max.depth = 7
                                              , min_child_weight = 100
                                              , subsample = 1
                                              , colsample_bytree = 1
                                              , nthread = 3
                                              , eval_metric = "auc"
                                              , tree_method = "hist"
                                              , grow_policy = "lossguide"
                              )
                              , watchlist = list(test = xgboost.testing)
                              , nrounds = 500
                              , early_stopping_rounds = 40
                              , print_every_n = 20
  )
  , times = 5L
)
)
#user  system elapsed 
#1.76    0.58    1.67 

# Relative importance of model attributes
xgb.importance(model = xgb)
xgb.plot.importance(importance_matrix = xgb.importance( model = xgb))
############################################################################################
# Model performance specific statistics on training data
library(ROCR)
p1.training.output.xgb = predict(xgb, xgboost.training, type = "prob")

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.xgb)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.xgb)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.xgb)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                p1_prediction_df=p1.training.output.xgb)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                                     p1_prediction_df=p1.training.output.xgb,
                                     threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
p1.testing.output.xgb = predict(xgb, xgboost.testing, type = "prob")

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.xgb)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.xgb)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.xgb)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.xgb)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.xgb,
                                     threshold=0.5)
############################################################################################
############################################################################################

############################################################################################
############################################################################################
# Neural Networks without using H2O
library(neuralnet)
set.seed(333)

system.time(
  nnet <- neuralnet(formula = formula.best.lasso,
                    data = input.table.dummy.normalized.training,
                    hidden = c(3),
                    err.fct = "ce",
                    rep = 1,
                    stepmax=1e6,
                    algorithm = "rprop+",
                    act.fct = "logistic", 
                    linear.output = FALSE)
)
#user  system elapsed 
#320.69    0.93  323.86 
plot(nnet)

# Relative importance of model attributes
#require(devtools)
#import 'gar.fun' from Github
#source_gist('6206737')
library(reshape)
source("C:/Data_Science/Training/MachineLearningWithR/gar.fun.R")
gar.fun(input.table.dummy.normalized.training,nnet)

library(NeuralNetTools)
garson(nnet)
olden(nnet)
############################################################################################
# Model performance for training data
library(ROCR)
net.output.training = compute(nnet, input.table.dummy.normalized.training[,  !names(input.table.dummy.normalized.training) %in% target_variable_name])
p1.training.output.nnet = net.output.training$net.result[,2]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.training, 
                 p1_prediction_df=p1.training.output.nnet)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.training, 
                        p1_prediction_df=p1.training.output.nnet)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.training, 
                    p1_prediction_df=p1.training.output.nnet)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.training, 
                p1_prediction_df=p1.training.output.nnet)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.training, 
                                     p1_prediction_df=p1.training.output.nnet,
                                     threshold=0.5)
############################################################################################
# Model performance for test data
library(ROCR)
net.output.testing = compute(nnet, input.table.dummy.normalized.testing[,  !names(input.table.dummy.normalized.testing) %in% target_variable_name])
p1.testing.output.nnet = net.output.testing$net.result[,2]

# Find the Gini coefficient
Gini_coefficient(input_table_name_df=input.table.dummy.normalized.testing, 
                 p1_prediction_df=p1.testing.output.nnet)
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots(input_table_name_df=input.table.dummy.normalized.testing, 
                        p1_prediction_df=p1.testing.output.nnet)
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x(input_table_name_df=input.table.dummy.normalized.testing, 
                    p1_prediction_df=p1.testing.output.nnet)
# Find optimum cut-offs
optimum_cutoffs(input_table_name_df=input.table.dummy.normalized.testing, 
                p1_prediction_df=p1.testing.output.nnet)
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff(input_table_name_df=input.table.dummy.normalized.testing, 
                                     p1_prediction_df=p1.testing.output.nnet,
                                     threshold=0.5)
############################################################################################
############################################################################################



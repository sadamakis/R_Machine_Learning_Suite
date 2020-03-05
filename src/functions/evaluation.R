############################################################################################
############################################################################################
## This code calculate functions that will be used for model evaluation
############################################################################################
############################################################################################

############################################################################################
############################################################################################
# ROC curve and accuracy, sensitivity, specificity vs cut-off plots
model_performance_plots = function(
  input_table_name_df, # The dataframe that has the data to produce the model performance
  p1_prediction_df # The dataframe that has the model predictions
)
{
  library(ROCR)
  #  par(mfrow=c(3,2))
  par(mfrow=c(3,2), mar=c(4,4,4,4))
  prediction.output = ROCR::prediction(p1_prediction_df, input_table_name_df[, names(input_table_name_df)==target_variable_name])
  performance.output.ROC = performance(prediction.output, "tpr", "fpr", cost.fp=1, cost.fn=1)
  plot(performance.output.ROC, main="ROC curve", colorize=T)
  abline(a=0, b=1)
  performance.output.prec.rec = performance(prediction.output, "prec", "rec", cost.fp=1, cost.fn=1)
  plot(performance.output.prec.rec, main="Precision/recall (PPV/Cum. ADR) curve", colorize=T, xlim=c(0,1), ylim=c(0,1))
  abline(a=1, b=-1)
  performance.output.accuracy = performance(prediction.output, "acc", cost.fp=1, cost.fn=1)
  plot(performance.output.accuracy, main="Accuracy vs cutoff")
  performance.output.sensitivity = performance(prediction.output, "sens", cost.fp=1, cost.fn=1)
  plot(performance.output.sensitivity, main="Sensitivity (Cum. ADR/TPR) vs cutoff")
  performance.output.specificity = performance(prediction.output, "rec", cost.fp=1, cost.fn=1)
  plot(performance.output.specificity, main="Specificity (TNR) vs cutoff")
  performance.output.fpr = prediction.output@fp[[1]]/prediction.output@tp[[1]]
  plot(prediction.output@cutoffs[[1]], performance.output.fpr, main="Cum. False Postive Rate (Cum. FPR) vs cutoff",  type='l', ylab='FPR', xlab='Cutoff')
  par(mfrow=c(1,1))
}

############################################################################################
############################################################################################
# Find optimum cut-offs
optimum_cutoffs = function(
  input_table_name_df, # The dataframe that has the data to produce the model performance
  p1_prediction_df # The dataframe that has the model predictions
)
{
  library(ROCR)
  prediction.output = ROCR::prediction(p1_prediction_df, input_table_name_df[, names(input_table_name_df)==target_variable_name])
  performance.output.accuracy = performance(prediction.output, "acc", cost.fp=1, cost.fn=1)
  performance.output.sensitivity = performance(prediction.output, "sens", cost.fp=1, cost.fn=1)
  performance.output.specificity = performance(prediction.output, "rec", cost.fp=1, cost.fn=1)
  
  optimum.cutoff.training.accuracy = slot(performance.output.accuracy, "x.values")[[1]][which.max(slot(performance.output.accuracy, "y.values")[[1]])]
  optimum.cutoff.training.sensitivity = slot(performance.output.sensitivity, "x.values")[[1]][which.max(slot(performance.output.sensitivity, "y.values")[[1]])]
  optimum.cutoff.training.specificity = slot(performance.output.specificity, "x.values")[[1]][which.max(slot(performance.output.specificity, "y.values")[[1]])]
  cat("Optimum cut-off for accuracy [(TP+TN)/(P+N)] = ", optimum.cutoff.training.accuracy)
  cat("\n")
  cat("Optimum cut-off for sensitivity [TPR = TP/P)] = ", optimum.cutoff.training.sensitivity)
  cat("\n")
  cat("Optimum cut-off for specificity [TNR = TN/N] = ", optimum.cutoff.training.specificity)
  cat("\n")
}

############################################################################################
############################################################################################
# Find the Gini coefficient
Gini_coefficient = function(
  input_table_name_df, # The dataframe that has the data to produce the model performance
  p1_prediction_df # The dataframe that has the model predictions
)
{
  library(ModelMetrics)
  Gini = 2*auc(input_table_name_df[, names(input_table_name_df)==target_variable_name], p1_prediction_df) - 1
  cat("Gini = ", Gini)
  cat("\n")

#library(WeightedROC)
#tp.fp <- WeightedROC(p1_prediction_df, input_table_name_df[, names(input_table_name_df)==target_variable_name], input_table_name_df[, names(input_table_name_df)==weight_variable_name])
#2*WeightedAUC(tp.fp)-1

}

############################################################################################
############################################################################################
# ADR/FPR vs population distribution, and lifting table
ADR_FPR_plots_top_x = function(
  input_table_name_df, # The dataframe that has the data to produce the model performance
  p1_prediction_df # The dataframe that has the model predictions
)
{
  library(ROCR)
  prediction.output = ROCR::prediction(p1_prediction_df, input_table_name_df[, names(input_table_name_df)==target_variable_name])
  performance.output.ROC = performance(prediction.output, "tpr", "fpr", cost.fp=1, cost.fn=1)
  performance.output.prec.rec = performance(prediction.output, "prec", "rec", cost.fp=1, cost.fn=1)
  
  Lifting_table = data.frame(performance.output.prec.rec@alpha.values[[1]], 
                             prediction.output@n.pos.pred[[1]]/length(p1_prediction_df),
                             prediction.output@fp[[1]]/prediction.output@tp[[1]],
                             performance.output.ROC@y.values[[1]])[-1,]
  colnames(Lifting_table) = c('Score cutoff', 
                              'Pop. Distribution', 
                              'Cum. FPR', 
                              'Cum. ADR')
  par(mfrow=c(1,2))
  # Plot ADR and FPR
  plot(Lifting_table[,names(Lifting_table)=='Pop. Distribution'], Lifting_table[,names(Lifting_table)=='Cum. ADR'], type='l', main="Cum. ADR vs Population Distribution", ylab='Cum. ADR', xlab='Population Distribution')
  plot(Lifting_table[,names(Lifting_table)=='Pop. Distribution'], Lifting_table[,names(Lifting_table)=='Cum. FPR'], type='l', main="Cum. FPR vs Population Distribution", ylab='Cum. FPR', xlab='Population Distribution')
  par(mfrow=c(1,1))
  # Detect top 5% and top 10%
  cat("Lifting Table")
  cat("\n")
  return(
    rbind(
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.05) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.05))), ], 
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.10) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.10))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.15) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.15))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.20) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.20))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.25) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.25))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.30) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.30))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.35) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.35))), ],
      Lifting_table[which(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.40) == min(abs(Lifting_table[,names(Lifting_table)=='Pop. Distribution']-0.40))), ]
    )
  )
}

############################################################################################
############################################################################################
# Confusion matrix for specific cutoffs
confusion_matrix_for_specific_cutoff = function(
  input_table_name_df, # The dataframe that has the data to produce the model performance
  p1_prediction_df, # The dataframe that has the model predictions
  threshold # The threshold value above which a positive outcome is predicted
)
{
  pred1 = ifelse(p1_prediction_df>threshold, 1, 0)
  library(caret)
  library(e1071)
  confusion_matrix = caret::confusionMatrix(as.factor(pred1), as.factor(input_table_name_df[, names(input_table_name_df)==target_variable_name]), positive = "1")
  accuracy.output = confusion_matrix$overall[which(names(confusion_matrix$overall)=="Accuracy")]
  sensitivity.output = confusion_matrix$byClass[which(names(confusion_matrix$byClass)=="Sensitivity")]
  specificity.output = confusion_matrix$byClass[which(names(confusion_matrix$byClass)=="Specificity")]
  
  cat("Accuracy [(TP+TN)/(P+N)] for threshold", threshold, "= ", accuracy.output)
  cat("\n")
  cat("Specificity [TNR = TN/N] for threshold", threshold, "= ", specificity.output)
  cat("\n")
  cat("Sensitivity [TPR = TP/P] for threshold", threshold, "= ", sensitivity.output)
  cat("\n")
  
  cat("The confusion matrix for threshold", threshold, "is:")
  cat("\n")
  return(
    confusion_matrix$table
  )
}
############################################################################################
############################################################################################


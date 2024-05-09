#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Objective Objective: The primary objective of implementing machine learning (ML) in the detection of cardiovascular disease
# is to enhance early and accurate identification of individuals at risk, enabling timely intervention and personalized 
# healthcare. Through the utilization of advanced ML algorithms on diverse patient data, including clinical records,
# medical imaging, and genetic information, we aim to develop a robust and efficient system that can predict and classify 
# cardiovascular diseases with high sensitivity and specificity. This implementation seeks to contribute to the improvement of
# patient outcomes by facilitating proactive healthcare strategies, optimizing resource allocation, and ultimately reducing
# the burden of cardiovascular diseases on both individuals and healthcare systems.


# In[37]:


#Importing necessary libraries for ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Cardiovascular_Disease_Dataset.csv')


# In[38]:


# Printing the dataset's head values/column/labels along with their values 

print(data.head())


# # Data Pre-Processing

# In[39]:


# Separating features (X) and target variable (y)
X = data[['patientid', 'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[40]:


# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train[['patientid', 'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']])
X_test = scaler.transform(X_test[['patientid', 'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']])


# # Training the Models

# In[41]:


# Training the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[42]:


# Training the random forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[43]:


# Training the support vector machine model
svm = SVC(probability=True)

svm.fit(X_train, y_train)


# In[44]:


# Evaluating the models on the test set
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)


# In[45]:


# Calculating the accuracy scores
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svm = accuracy_score(y_test, y_pred_svm)


# In[46]:


# Printing the accuracy scores
print('Accuracy score for logistic regression:', acc_lr)
print('Accuracy score for random forest:', acc_rf)
print('Accuracy score for support vector machine:', acc_svm)


# # Histograms for Data Analysis

# In[47]:


# Creating histograms and graphs for analyzing each model's working


# In[48]:


# Creating separate histograms for positive and negative predictions
plt.figure(figsize=(10, 5))

# Positive class predictions
y_pred_lr_pos = y_pred_lr[y_test == 1]
plt.hist(y_pred_lr_pos, bins=20, alpha=0.5, label="Positive class")

# Negative class predictions
y_pred_lr_neg = y_pred_lr[y_test == 0]
plt.hist(y_pred_lr_neg, bins=20, alpha=0.5, label="Negative class")

# Plot formatting
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Logistic regression prediction distribution')
plt.legend()
plt.show()


# In[49]:


# Creating separate histograms for positive and negative predictions
plt.figure(figsize=(10, 5))

# Positive class predictions
y_pred_rf_pos = y_pred_rf[y_test == 1]
plt.hist(y_pred_rf_pos, bins=20, alpha=0.5, label="Positive class")

# Negative class predictions
y_pred_rf_neg = y_pred_rf[y_test == 0]
plt.hist(y_pred_rf_neg, bins=20, alpha=0.5, label="Negative class")

# Plot formatting
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Random forest prediction distribution')
plt.legend()
plt.show()


# In[50]:


# Creating separate histograms for positive and negative predictions
plt.figure(figsize=(10, 5))

# Positive class predictions
y_pred_svm_pos = y_pred_svm[y_test == 1]
plt.hist(y_pred_svm_pos, bins=20, alpha=0.5, label="Positive class")

# Negative class predictions
y_pred_svm_neg = y_pred_svm[y_test == 0]
plt.hist(y_pred_svm_neg, bins=20, alpha=0.5, label="Negative class")

# Plot formatting
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Support vector machine prediction distribution')
plt.legend()
plt.show()


# # Classification Report & Accuracy Analysis

# In[51]:


# Classification report
print('Classification report for logistic regression:')
print(classification_report(y_test, y_pred_lr))


# In[52]:


print('Classification report for random forest:')
print(classification_report(y_test, y_pred_rf))


# In[53]:


print('Classification report for support vector machine:')
print(classification_report(y_test, y_pred_svm))


# # Determination of the best over Model 

# In[54]:


# As the accuracy of the random forest model and simple vector machine are better than logistic
# regression model i.e 98%,98% vs 96%, therefore the most accurate models random forest and svm may be used ,although
# the over all accuracy of the models are exceptional for implemenetaion i.e out of 100 tests, 95 of them will be 
# successful and only 5 +- will maybe be predicted as false positive prediction(s).



# In[55]:


# However, in terms of the best model,its one that 
# balances accuracy,generalizability, interpretability, computational cost, robustness, calibration, and 
# other relevant factors according to your specific needs and priorities.
# this can be analyzed by ROC curves,Precision-recall curves,Scatter plotsas implemented below.


# In[56]:


# Logistic Regression
y_proba_lr = lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_test, y_proba_lr)

# Random Forest
y_proba_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_proba_rf)

# Support Vector Machine
y_proba_svm = svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_proba_svm)
precision_svm, recall_svm, thresholds_svm = precision_recall_curve(y_test, y_proba_svm)

# ROC curves
plt.figure(figsize=(10, 5))
plt.plot(fpr_lr, tpr_lr, label="Logistic regression", color="blue")
plt.plot(fpr_rf, tpr_rf, label="Random forest", color="green")
plt.plot(fpr_svm, tpr_svm, label="Support vector machine", color="red")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
plt.show()

# Precision-recall curves
plt.figure(figsize=(10, 5))
plt.plot(precision_lr, recall_lr, label="Logistic regression", color="blue")
plt.plot(precision_rf, recall_rf, label="Random forest", color="green")
plt.plot(precision_svm, recall_svm, label="Support vector machine", color="red")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision-recall curves")
plt.legend()
plt.show()

# Scatter plots (predicted probability vs. true label)
plt.figure(figsize=(10, 5))
plt.scatter(y_proba_lr, y_test, label="Logistic regression", color="blue")
plt.scatter(y_proba_rf, y_test, label="Random forest", color="green")
plt.scatter(y_proba_svm, y_test, label="Support vector machine", color="red")
plt.xlabel("Predicted probability of cardiovascular disease")
plt.ylabel("True class label")
plt.title("Scatter plots")
plt.legend()
plt.show()


# In[57]:


# the Random Forest model appears to be the best overall model for predicting cardiovascular disease.
# It has the highest ROC curve (area under the grapgh/curve to be precise), the highest area under the precision-recall curve, and the most separation 
# between the positive and negative classes in the scatter plot. However, data sets also are a variable that may
# affect the models accuracy overall. 


# # Testing the model on new dataset

# In[58]:


new_data = pd.read_csv('ftestheart.csv')
X_new_test = scaler.transform(new_data[['patientid', 'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']])
y_pred_lr_new = lr.predict(X_new_test)
y_pred_rf_new = rf.predict(X_new_test)
y_pred_svm_new = svm.predict(X_new_test)


# In[59]:


# Create histograms for positive and negative predictions
plt.figure(figsize=(15, 10))

# Logistic Regression
plt.subplot(3, 1, 1)
plt.hist(y_pred_lr_pos, bins=20, alpha=0.5, label="Positive class (Logistic Regression)")
plt.hist(y_pred_lr_neg, bins=20, alpha=0.5, label="Negative class (Logistic Regression)")
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Prediction Distribution for Logistic Regression')
plt.legend()

# Random Forest
plt.subplot(3, 1, 2)
plt.hist(y_pred_rf_pos, bins=20, alpha=0.5, label="Positive class (Random Forest)")
plt.hist(y_pred_rf_neg, bins=20, alpha=0.5, label="Negative class (Random Forest)")
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Prediction Distribution for Random Forest')
plt.legend()

# Support Vector Machine
plt.subplot(3, 1, 3)
plt.hist(y_pred_svm_pos, bins=20, alpha=0.5, label="Positive class (Support Vector Machine)")
plt.hist(y_pred_svm_neg, bins=20, alpha=0.5, label="Negative class (Support Vector Machine)")
plt.xlabel('Predicted probability of cardiovascular disease')
plt.ylabel('Number of samples')
plt.title('Prediction Distribution for Support Vector Machine')
plt.legend()

plt.tight_layout()
plt.show()

# Show the binary predictions (1 or 0) for each model
print('Binary predictions for logistic regression:')
print(y_pred_lr_new)

print('Binary predictions for random forest:')
print(y_pred_rf_new)

print('Binary predictions for support vector machine:')
print(y_pred_svm_new)


# In[60]:


#SVM is usually more prone to false readings if there are outliers ,which may be possible in this case .
#Overall, as tested previosly, the Random Forest model is the best over all model to determine the cardeovescular
#diseases . 



# In[63]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# Load the test data
test_data = pd.read_csv('testcardio.csv')

# Display the test data
print("Test Data:")
print(test_data.head())

# Load the trained Random Forest model
rf = RandomForestClassifier()
# You need to load the saved model from your training phase here
rf = load_model_function()

# Assuming you have a function to preprocess input features and fit the scaler
def preprocess_input(input_data, scaler=None):
    # Add your preprocessing steps here
    if scaler is None:
        scaler = StandardScaler()
        processed_input = scaler.fit_transform(input_data)
    else:
        processed_input = scaler.transform(input_data)
    return processed_input, scaler

# Function for real-time prediction using the trained Random Forest model
def predict_real_time(model, input_data, scaler=None):
    # Preprocess input data
    processed_input, scaler = preprocess_input(input_data, scaler)
    
    # Make predictions
    predictions = model.predict(processed_input)
    
    return predictions

# Example usage
# Assume 'test_data' is the DataFrame containing the test data
# Extract the features from the test data
X_test_real_time = test_data[['patientid', 'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']]

# Fit the model with training data before real-time prediction
# You need to load the appropriate training data and labels for fitting
X_train, y_train = load_training_data()
rf.fit(X_train, y_train)

# Assuming you have a model fitting function, use it here
fit_model_function(rf, X_train, y_train)

# Make real-time prediction
real_time_predictions = predict_real_time(rf, X_test_real_time)

# Display the real-time predictions
print("Real-time Predictions:")
print(real_time_predictions)


# In[ ]:





# In[ ]:





# In[ ]:





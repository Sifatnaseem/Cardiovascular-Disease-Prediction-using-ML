import pandas as pd
df = pd.read_csv ("C:\\Users\\Azmat\\OneDrive\\Desktop\\preprocessed_dataset1.csv")
df.head(5)


# In[15]:


X = df [['age','gender','height','weight','systolic','diastolic','cholesterol','glucose','smoke','alcohol','active','pulse_pressure']]
y = df ['cardiovascular_disease'].ravel()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=109)


# In[17]:


from sklearn.svm import SVC
clf = SVC(kernel='rbf',C=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[18]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

FEATURE ANALYSIS
# We imported the permutation_importance module from sklearn. The permutation_importance module is indicative of how much the model 
# depends on a particular feature.
feature_importance = permutation_importance(clf, X_test, y_test, n_jobs=-1)
feature_importance

# As seen above the 'importances_mean' key corresponds to the mean of importance of each feature
# We would like to sort the array to get the importances in an increasing order
# We can sort it using the sort() method, however we will use argsort() method which will sort the list of indices that would be present after sorting
# This argsort will be used to plot the features in an increasing order without actually sorting the data 
args = feature_importance.importances_mean.argsort()
args

# Plotting the feature importance using the seaborn library
plt.figure(figsize=(5,5))
sns.barplot(y=X_test.columns[args], x=feature_importance.importances_mean[args], estimator=sum, orient="h", color= "steelblue")
plt.xlabel('Importance')
plt.title('Feature Analysis for SVM Model')
plt.show()





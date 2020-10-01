#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression Modeling for Early Stage Diabetes Risk Prediction

# ## Part 2.1: Getting familiar with linear algebraic functions

# #### Tasks
# - Create matrix of size 10*10 with random integer numbers
# - Compute the following linear algebric operations on the matrix using built in functions supported in Numpy, Scipy etc.
#   - Find inverse of the matrix and print it
#   - Calculate dot product of the matrix with same matrix in transpose A.AT
#   - Decompose the original matrix using eigen decomposition print the eigen values and eigen vectors
#   - Calculate jacobian matrix 
#   - Calculate hessian matrix

# In[3]:


import numpy as np
random_matrix_array =np.random.randint(1,1000,size=(10,10))
random_matrix_array 


# In[4]:


rev = np.linalg.inv(random_matrix_array) 


# In[5]:


rev


# In[10]:


trans=np.transpose(random_matrix_array)


# In[11]:


trans


# In[13]:


res=np.dot(trans,random_matrix_array)
res


# In[15]:


eigenvalues, eigenvectors = np.linalg.eig(random_matrix_array)


# In[17]:


print("Eigen Values ", eigenvalues)


# In[18]:


print("Eigen Vectors ", eigenvectors)


# In[20]:


get_ipython().system('pip install numdifftools')


# In[21]:


import numdifftools as nd


# In[30]:


from numdifftools import Jacobian, Hessian
from scipy.optimize import minimize
def fun(x, a):
    return (x[0] - 1) **2 + (x[1] - a) **2

def fun_der(x, a):
    return Jacobian(lambda x: fun(x, a))(x).ravel()
def fun_hess(x, a):
    return Hessian(lambda x: fun(x, a))(x)
x0 = np.array([2, 0]) # initial guess
a = 2.5

res = minimize(fun, x0, args=(a,), method='dogleg', jac=fun_der, hess=fun_hess)
print(res)


# ## Part 2.2: Logistic Regression using newton method

# ### Logistic regression
# Logistic regression uses an equation as the representation, very much like linear regression.
# 
# Input values (x) are combined linearly using weights or coefficient values (referred to as W) to predict an output value (y). A key difference from linear regression is that the output value being modeled is a binary values (0 or 1) rather than a continuous value.<br>
# 
# ###  $\hat{y}(w, x) = \frac{1}{1+exp^{-(w_0 + w_1 * x_1 + ... + w_p * x_p)}}$
# 
# #### Dataset
# The dataset is available at <strong>"data/diabetes_data.csv"</strong> in the respective challenge's repo.<br>
# <strong>Original Source:</strong> http://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv. The dataset just got released in July 2020.<br><br>
# 
# #### Features (X)
# 
# 1. Age                - Values ranging from 16-90
# 2. Gender             - Binary value (Male/Female)
# 3. Polyuria           - Binary value (Yes/No)
# 4. Polydipsia         - Binary value (Yes/No)
# 5. sudden weight loss - Binary value (Yes/No)
# 6. weakness           - Binary value (Yes/No)
# 7. Polyphagia         - Binary value (Yes/No)
# 8. Genital thrush     - Binary value (Yes/No)
# 9. visual blurring    - Binary value (Yes/No)
# 10. Itching           - Binary value (Yes/No)
# 11. Irritability      - Binary value (Yes/No)
# 12. delayed healing   - Binary value (Yes/No)
# 13. partial paresis   - Binary value (Yes/No)
# 14. muscle stiffness  - Binary value (Yes/No)
# 15. Alopecia          - Binary value (Yes/No)
# 16. Obesity           - Binary value (Yes/No)
# 
# #### Output/Target target (Y) 
# 17. class - Binary class (Positive/Negative)
# 
# #### Objective
# To learn logistic regression and practice handling of both numerical and categorical features
# 
# #### Tasks
# - Download, load the data and print first 5 and last 5 rows
# - Transform categorical features into numerical features. Use label encoding or any other suitable preprocessing technique
# - Since the age feature is in larger range, age column can be normalized into smaller scale (like 0 to 1) using different methods such as scaling, standardizing or any other suitable preprocessing technique (Example - sklearn.preprocessing.MinMaxScaler class)
# - Define X matrix (independent features) and y vector (target feature)
# - Split the dataset into 60% for training and rest 40% for testing (sklearn.model_selection.train_test_split function)
# - Train Logistic Regression Model on the training set (sklearn.linear_model.LogisticRegression class)
# - Use the trained model to predict on testing set
# - Print 'Accuracy' obtained on the testing dataset i.e. (sklearn.metrics.accuracy_score function)
# 
# #### Further fun (will not be evaluated)
# - Plot loss curve (Loss vs number of iterations)
# - Preprocess data with different feature scaling methods (i.e. scaling, normalization, standardization, etc) and observe accuracies on both X_train and X_test
# - Training model on different train-test splits such as 60-40, 50-50, 70-30, 80-20, 90-10, 95-5 etc. and observe accuracies on both X_train and X_test
# - Shuffling of training samples with different *random seed values* in the train_test_split function. Check the model error for the testing data for each setup.
# - Print other classification metrics such as:
#     - classification report (sklearn.metrics.classification_report),
#     - confusion matrix (sklearn.metrics.confusion_matrix),
#     - precision, recall and f1 scores (sklearn.metrics.precision_recall_fscore_support)
# 
# #### Helpful links
# - Scikit-learn documentation for logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# - How Logistic Regression works: https://machinelearningmastery.com/logistic-regression-for-machine-learning/
# - Feature Scaling: https://scikit-learn.org/stable/modules/preprocessing.html
# - Training testing splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# - Classification metrics in sklearn: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# - Use slack for doubts: https://join.slack.com/t/deepconnectai/shared_invite/zt-givlfnf6-~cn3SQ43k0BGDrG9_YOn4g

# In[31]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


# Download the dataset from the source
# !wget _URL_


# In[46]:


# NOTE: DO NOT CHANGE THE VARIABLE NAME(S) IN THIS CELL
# Load the data
data = pd.read_csv("data/diabetes_data.csv")


# In[47]:


data.head()


# In[48]:


data.tail()


# In[49]:


# Handle categorical/binary columns


# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[51]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[52]:


cols = list(data.columns)


# In[53]:


cols


# In[59]:


for i in cols :
    data[i]=(le.fit_transform(data[i]))


# In[60]:


data.head()


# In[ ]:


# Normalize the age feature


# In[66]:


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(data)
data = min_max_scaler.transform(data)
data


# In[69]:


data[:,:-1]


# In[70]:


data[:,-1]


# In[71]:


# Define your X and y
X =  data[:,:-1]
y =  data[:,-1]


# In[72]:


# Split the dataset into training and testing here
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,train_size=.6)


# In[73]:


def predict(X, weights):
    '''Predict class for X.
    For the given dataset, predicted vector has only values 0/1
    Args:
        X : Numpy array (num_samples, num_features)
        weights : Model weights for logistic regression
    Returns:
        Binary predictions : (num_samples,)
    '''

    ### START CODE HERE ###
    z = np.dot(X,weights)
    logits = sigmoid(z)
    y_pred = np.array(list(map(lambda x: 1 if x>0.5 else 0, logits)))
    ### END CODE HERE ###
    
    return y_pred


# In[74]:


def sigmoid(z):
        '''Sigmoid function: f:R->(0,1)
        Args:
            z : A numpy array (num_samples,)
        Returns:
            A numpy array where sigmoid function applied to every element
        '''
        ### START CODE HERE
        sig_z = 1/(1+ np.exp(-z))
        ### END CODE HERE
        
        assert (z.shape==sig_z.shape), 'Error in sigmoid implementation. Check carefully'
        return sig_z


# In[75]:


def cross_entropy_loss(y_true, y_pred):
    '''Calculate cross entropy loss
    Note: Cross entropy is defined for multiple classes/labels as well
    but for this dataset we only need binary cross entropy loss
    Args:
        y_true : Numpy array of true values (0/1) of size (num_samples,)
        y_pred : Numpy array of predicted values (probabilites) of size (num_samples,)
    Returns:
        Cross entropy loss: A scalar value
    '''
    # Fix 0 values in y_pred
    y_pred = np.maximum(np.full(y_pred.shape, 1e-7), np.minimum(np.full(y_pred.shape, 1-1e-7), y_pred))
    
    ### START CODE HERE
    ce_loss = np.mean(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred))
    ### END CODE HERE
    
    return ce_loss


# In[76]:


def newton_optimization(X, y, max_iterations=25):
    '''Implement netwon method for optimizing weights
    Args:
        X : Numpy array (num_samples, num_features)
        max_iterations : Max iterations to update the weights
    Returns:
        Optimal weights (num_features,)
    '''
    num_samples = X.shape[0]
    num_features = X.shape[1]
    # Initialize random weights
    weights = np.zeros(num_features,)
    # Initialize losses
    losses = []
    
    # Newton Method
    for i in range(max_iterations):
        # Predict/Calculate probabilties using sigmoid function
        z=np.dot(X,weights)
        y_p = sigmoid(z)
        
        # Define gradient for J (cost function) i.e. cross entropy loss
        gradient = 1./num_samples*np.dot(X.T,(y_p-y))
        
        # Define hessian matrix for cross entropy loss
        hessian = 1./num_samples*X.T.dot(np.diag(y_p*(1-y_p))).dot(X)
        
        # Update the model using hessian matrix and gradient computed
        weights-= np.dot(np.linalg.pinv(hessian),gradient) 
        
        # Calculate cross entropy loss
        loss = cross_entropy_loss(y, y_p)
        # Append it
        losses.append(loss)

    return weights, losses


# In[77]:


# Train weights
weights, losses = newton_optimization(X_train, y_train)


# In[78]:


# Plot the loss curve
plt.plot([i+1 for i in range(len(losses))], losses)
plt.title("Loss curve")
plt.xlabel("Iteration num")
plt.ylabel("Cross entropy curve")
plt.show()


# In[79]:


our_model_test_acuracy = accuracy_score(y_test, predict(X_test, weights))

print(f"\nAccuracy in testing set by our model: {our_model_test_acuracy}")


# #### Compare with the scikit learn implementation

# In[80]:


# Initialize the model
model = LogisticRegression(solver='newton-cg', verbose=1)


# In[81]:


# Fit the model. Wait! We will complete this step for you ;)
model.fit(X_train, y_train)


# In[82]:


# Predict on testing set X_test
y_pred = model.predict(X_test)


# In[83]:


# Print Accuracy on testing set
sklearn_test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy in testing set by sklearn model: {sklearn_test_accuracy}")


# In[ ]:





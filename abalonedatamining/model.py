#assessment 3 code: 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random

import seaborn as sns
from numpy import float32
#Now to clean the data: 
data = pd.read_csv('abalone.data', sep=',', header = None)
print(data)
in_data = data.values
in_data[:,0] = np.where(in_data[:,0] == 'M', 0, np.where(in_data[:,0] == 'F', 1, -1)) #relabelling gender to numerical values for better processing. 

in_data = in_data.astype(float32) #making sure the values are float type so correlation matrix can be created
corr_matr = np.corrcoef(in_data.T) #transposing the array
print(corr_matr, ' is the correlation matrix of the data')

data=data.replace(['M','F','I'],[0,1,-1]) #experimenting with an easier way to replace the gender letters with numbers

data=np.array(data)

candidates = {'Sex': data[:,0],
             'Length': data[:,1],
             'Diameter': data[:,2],
             'Height': data[:,3],
             'Whole_weight': data[:,4],
             'Shucked_weight': data[:,5], #Weight of the abalone without shell
             'Viscera_weight':data[:,6],  #weight of the abalone internal organs
             'Shell_weight':data[:,7],
             'Rings':data[:,8]}

df = pd.DataFrame(candidates, columns= ['Sex','Length','Diameter','Height','Whole_weight',
                                        'Shucked_weight','Viscera_weight','Shell_weight','Rings'])
for i in range(len(df['Rings'])):  #grouping the ages for convenience later on
    if df['Rings'][i]<=7:
        df['Rings'][i]=1
    if df['Rings'][i]<=10 and df['Rings'][i]>=8:
        df['Rings'][i]=2
    if df['Rings'][i]<=15 and df['Rings'][i]>=11:
        df['Rings'][i]=3
    if df['Rings'][i]>15:
        df['Rings'][i]=4

def visualise_heatmap(data):  #now to generate the heatmap using correlation matrix. 
    correlation_matrix = df.corr().round(2) #choosing two decimal places for ease to read output
    print(correlation_matrix)
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.rc('figure',figsize=(8,8))
    plt.savefig('heatmap.png')
    plt.show()
    
visualise_heatmap(df) #heatmap shown in report

#Now to begin building the decision tree algorithm: 

df.Rings = df['Rings'].apply(int)
df.Rings = df['Rings'].apply(str) #converting integer to string 

X = df[['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']]
Y = df['Rings']  #This is the variable to be predicted 

def create_set(n):
    from sklearn.model_selection import train_test_split

    X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.4, random_state=n)  #Now randomly splitting the set in to 60% for training and 40% for testing
    print(X_training.shape)
    print(X_testing.shape)
    print(Y_training.shape)
    print(Y_testing.shape)
    
    return X_training, X_testing, Y_training, Y_testing

X_training, X_testing, Y_training, Y_testing=create_set(10)  #note that this was changed to 25 for getting the other decision tree shown in report. 

#Now to Import the relevant packages from sklearn and use the training set 
from sklearn.tree import DecisionTreeClassifier

x = X_training
y = Y_training

tree_classf = DecisionTreeClassifier(random_state=0, max_depth=2, criterion = 'gini') #implementing the Gini impurity measure:
tree_classf.fit(x, y)  

#Now to label the attributes to used in training:

feature_name = ['Sex','Length','Diameter','Height','Whole_weight',
                'Shucked_weight','Viscera_weight','Shell_weight']

target_name = ['Rings']

tree_classf = DecisionTreeClassifier(random_state=0, max_depth=2, criterion = 'gini')
tree_classf.fit(x, y)

from sklearn.tree import export_graphviz

export_graphviz(
                tree_classf,
                out_file='tree1.dot', 
                feature_names = df.columns[:8],
                class_names = df.Rings,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
#Now to convert .dot to png: 

import pydot

(graph,) = pydot.graph_from_dot_file('tree1.dot')
graph.write_png('tree1.png')


#Now to plot the tree: 
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree1.png'))
plt.axis('off');
plt.savefig('CART_dtree1.png')
plt.show() 

#Next step is pruning the tree: 

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_training, Y_training)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
plt.rc('figure',figsize=(5,5))
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig('impurity_vsalpha1.png')
plt.show()  


classfs = []
for ccp_alpha in ccp_alphas:
    classf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    classf.fit(X_training, Y_training)
    classfs.append(classf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      classfs[-1].tree_.node_count, ccp_alphas[-1]))  #get rid of trivial tree with just one node. 


classfs = classfs[:-1]
ccp_alphas = ccp_alphas[:-1] #removing the last element (one node tree)

node_counts = [classf.tree_.node_count for classf in classfs]
depth = [classf.tree_.max_depth for classf in classfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of the tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig('nodes_vs_alpha.png')

#Now to see accuracy and alpha scores across training and testing sets:
training_scores = [classf.score(X_training, Y_training) for classf in classfs]
testing_scores = [classf.score(X_testing, Y_testing) for classf in classfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, training_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, testing_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig('xxx.png')

plt.show()

#To generate the optimal decision with 5 layers:
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth = 5, n_estimators=10)

model.fit(x, y)
estimator = model.estimators_[5]
from sklearn.tree import export_graphviz
# Export dot file
export_graphviz(estimator, out_file='tree2.dot', 
                feature_names = df.columns[:8],
                class_names = df.Rings,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

#Now convert to PNG like above: 
import pydot

(graph,) = pydot.graph_from_dot_file('tree2.dot')
graph.write_png('tree2.png')


#Now to plot the tree: 
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree2.png'))
plt.axis('off');
plt.savefig('CART_dtree2.png')
plt.show()  

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random

import seaborn as sns
from numpy import float32
#Now to clean the data: 
data = pd.read_csv('abalone.data', sep=',', header = None)
print(data)
in_data = data.values
in_data[:,0] = np.where(in_data[:,0] == 'M', 0, np.where(in_data[:,0] == 'F', 1, -1)) #relabelling gender to numerical values for better processing. 

in_data = in_data.astype(float32) #making sure the values are float type so correlation matrix can be created
corr_matr = np.corrcoef(in_data.T) #transposing the array
print(corr_matr, ' is the correlation matrix of the data')

data=data.replace(['M','F','I'],[0,1,-1]) #experimenting with an easier way to replace the gender letters with numbers

data=np.array(data)

candidates = {'Sex': data[:,0],
             'Length': data[:,1],
             'Diameter': data[:,2],
             'Height': data[:,3],
             'Whole_weight': data[:,4],
             'Shucked_weight': data[:,5], #Weight of the abalone without shell
             'Viscera_weight':data[:,6],  #weight of the abalone internal organs
             'Shell_weight':data[:,7],
             'Rings':data[:,8]}

df = pd.DataFrame(candidates, columns= ['Sex','Length','Diameter','Height','Whole_weight',
                                        'Shucked_weight','Viscera_weight','Shell_weight','Rings'])
for i in range(len(df['Rings'])):  #grouping the ages for convenience later on
    if df['Rings'][i]<=7:
        df['Rings'][i]=1
    if df['Rings'][i]<=10 and df['Rings'][i]>=8:
        df['Rings'][i]=2
    if df['Rings'][i]<=15 and df['Rings'][i]>=11:
        df['Rings'][i]=3
    if df['Rings'][i]>15:
        df['Rings'][i]=4

def visualise_heatmap(data):  #now to generate the heatmap using correlation matrix. 
    correlation_matrix = df.corr().round(2) #choosing two decimal places for ease to read output
    print(correlation_matrix)
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.rc('figure',figsize=(8,8))
    plt.savefig('heatmap.png')
    plt.show()
    
visualise_heatmap(df) #heatmap shown in report

#Now to begin building the decision tree algorithm: 

df.Rings = df['Rings'].apply(int)
df.Rings = df['Rings'].apply(str) #converting integer to string 

X = df[['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight']]
Y = df['Rings']  #This is the variable to be predicted 

def create_set(n):
    from sklearn.model_selection import train_test_split

    X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = 0.4, random_state=n)  #Now randomly splitting the set in to 60% for training and 40% for testing
    print(X_training.shape)
    print(X_testing.shape)
    print(Y_training.shape)
    print(Y_testing.shape)
    
    return X_training, X_testing, Y_training, Y_testing

X_training, X_testing, Y_training, Y_testing=create_set(10)  #note that this was changed to 25 for getting the other decision tree shown in report. 

#Now to Import the relevant packages from sklearn and use the training set 
from sklearn.tree import DecisionTreeClassifier

x = X_training
y = Y_training

tree_classf = DecisionTreeClassifier(random_state=0, max_depth=2, criterion = 'gini') #implementing the Gini impurity measure:
tree_classf.fit(x, y)  

#Now to label the attributes to used in training:

feature_name = ['Sex','Length','Diameter','Height','Whole_weight',
                'Shucked_weight','Viscera_weight','Shell_weight']

target_name = ['Rings']

tree_classf = DecisionTreeClassifier(random_state=0, max_depth=2, criterion = 'gini')
tree_classf.fit(x, y)

from sklearn.tree import export_graphviz

export_graphviz(
                tree_classf,
                out_file='tree1.dot', 
                feature_names = df.columns[:8],
                class_names = df.Rings,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
#Now to convert .dot to png: 

import pydot

(graph,) = pydot.graph_from_dot_file('tree1.dot')
graph.write_png('tree1.png')


#Now to plot the tree: 
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree1.png'))
plt.axis('off');
plt.savefig('CART_dtree1.png')
plt.show() 

#Next step is pruning the tree: 

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_training, Y_training)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
plt.rc('figure',figsize=(5,5))
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig('impurity_vsalpha1.png')
plt.show()  


classfs = []
for ccp_alpha in ccp_alphas:
    classf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    classf.fit(X_training, Y_training)
    classfs.append(classf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      classfs[-1].tree_.node_count, ccp_alphas[-1]))  #get rid of trivial tree with just one node. 


classfs = classfs[:-1]
ccp_alphas = ccp_alphas[:-1] #removing the last element (one node tree)

node_counts = [classf.tree_.node_count for classf in classfs]
depth = [classf.tree_.max_depth for classf in classfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of the tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig('nodes_vs_alpha.png')

#Now to see accuracy and alpha scores across training and testing sets:
training_scores = [classf.score(X_training, Y_training) for classf in classfs]
testing_scores = [classf.score(X_testing, Y_testing) for classf in classfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, training_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, testing_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig('alpha_accuracy.png')

plt.show() 

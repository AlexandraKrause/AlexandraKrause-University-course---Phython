"""
assignment
"""
from skimage.io import imsave, imread, imshow, show, imshow_collection
import pickle
import random

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.linear_model import LinearRegression


ex01 = True
ex02 = True
ex03 = True
ex04 = True

# first the data will be loaded.
#data for number one:
print("background")
with open("data/PAML_data/Q1_BG_dict.pkl", "rb") as sf:
    picture_background = pickle.load(sf)

    print (picture_background)


with open("data/PAML_data/Q1_Red_dict.pkl", "rb") as sf:
    picture_red = pickle.load(sf)

    print (picture_red)

print("yellow")

with open("data/PAML_data/Q1_Yellow_dict.pkl", "rb") as sf:
    picture_yellow = pickle.load(sf)

    print(picture_yellow)



#data for number two:
print("BG")

with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)

    print(picture_BG)


print("SP")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)

# look at the data of number two and visualise it
    print(picture_SP.keys())
#dict_keys(['train', 'validation', 'evaluation'])
#The sets in the dictionary are: dict_keys(['train', 'validation', 'evaluation'])
#The size of the data matrix X for each set is: (10000, 3) (5000, 3) (5000, 3): Many pictures especially in
#training data set, but also each 5000 in the validation and evaluation data sets
#he size of each entry is: (64, 64, 3) (64, 64, 3) (64, 64, 3): These are very big pictures, as visible

#imshow(picture_SP["train"] [1])
#show()
#imshow_collection(picture_SP["train"])
#show()
#imshow_collection(picture_SP["validation"])
#show()
#imshow(picture_SP["validation"] [1])
#show()
#imshow_collection(picture_SP["evaluation"])
#show()
for fname in ['data/PAML_data/Q1_BG_dict.pkl',  'data/PAML_data/Q1_Red_dict.pkl',  'data/PAML_data/Q1_Yellow_dict.pkl']:
	print("PAML_data/Q1_BG_dict.pkl'", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The size of the data matrix X for each set is:", data['train'].shape, data['validation'].shape, data['evaluation'].shape)


for fname in ['data/PAML_data/Q2_BG_dict.pkl',  'data/PAML_data/Q2_SP_dict.pkl']:
	print("data/PAML_data/Q2_BG_dict.pkl", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The number of entries for each set is:", len(data['train']), len(data['validation']), len(data['evaluation']))
		print("The size of each entry is:", data['train'][0].shape, data['validation'][0].shape, data['evaluation'][0].shape)

"""
#solution 8 would test and train with gausians...would be more efficient???

x = data['picture_background']
print(x.shape)
y = data['picture_background']
print(y.shape)

# Now split these into a training and testing and evaluation data set.
# not a very efficient solution
# 1. import random in your import area.
# 2. Decide on a training/testing split:
# 70% train red, 70% background, 30% test!
# First Split 50-50. Then second split 50-50!!! 25% training and evaluation
split = int( 0.5 * numrows )
# 3. Create a "traininglist" of randomly assigned indexes that will select data from
# both x and y. To do this we will use random.sample( A, B ), where A is the full list of
# numbers we can select from: range(0, 20640) in this case. And B is the number of
# values you are going to select out of A. Try it now
trainlist = random.sample( range( 0, numrows), split )
# Now we need to create the evaluation list.
# Basically you will just go through range( numrows ) and if an integer is not in
# trainlist you will put it in evallist. Hopefully you see that this is not very
# efficient..
testlist = []
for i in range(numrows):
    if i not in trainlist:
        testlist.append(i)
# Would this do the same thing:
# testlist1 = [i for i in range( numrows ) if i not in trainlist]
# if testlist == testlist1:
#   print( 'test lists equal' )
# So now we have a list of indexs that corrospond to the training and evaluation
# samples. Let's use these indexes on x and y to create the subsets.
x_train = x[trainlist]
y_train = y[trainlist]
print( x_train.shape, y_train.shape )
x_test = x[testlist]
y_test = y[testlist]
print( x_test.shape, y_test.shape )
# Now we have the data to train a model and the data to evaluate how good our model is.
# Let's plot these two sub sets individually.
plt.figure()
plt.scatter( x_train, y_train )
plt.xlabel( 'median income' )
plt.ylabel( 'median house value' )
plt.title( 'training data' )
plt.tight_layout()
plt.savefig( 'ex3training.pdf' )

plt.figure()
plt.scatter( x_test, y_test )
plt.xlabel( 'median income' )
plt.ylabel( 'median house value' )
plt.title( 'testing data' )
plt.tight_layout()
plt.savefig( 'ex3testing.pdf' )
print("red")

"""
###Number 1. kmeans
class KMeans_self():
  # Next we need to create the __init__ function that takes as input the number of
  # clusters (n_clusters), and the max iterations (imax) set to 100 as default.
  # We could add a distance metric here too, do you know what it would do?
  def __init__(self, n_clusters, imax=100):
    # instantiate the inputs
    self.n_clusters = n_clusters
    self.imax = imax

  # Now let's create a the Euchlidean distance calculator (euchlid) that takes some data (X)
  # and a center value (self.C[c]) as input. This is based on (sum( (X-C)^2 ))^(1/2.) where the resulting vector
  # will have the same number of columns as the input X.

  # alex: with the distance measure euclidean distance (L2), also called the sum of squared differences,
  # we will measure between centers. The manhattan distance would rather be for
  # sparse data, so here the decision is to first use the euclidean distance. The Manhattan distance is an
  # approximation to Euclidean distance and cheaper to compute
  # Translation invariant.
  def euchlid(self, X, c):
    diff = X - c
    sqrd = diff ** 2
    smmd = np.sum(sqrd, axis=1)
    return np.sqrt(smmd)

  # Next is the main part of the code, this is based on the algorithm in the pdf.
  # See if you can work it out from the sudo code supplied. But call the function "fit"
  def fit(self, X):
    # first we need to randomly create the cluster centers.
    # random dpoint selection
    cstart = np.random.randint( 0, X.shape[0], self.n_clusters )
    self.C = X[cstart,:]
    ### You could also do:
    #xmin = X.min(axis=0)
    #xmax = X.max(axis=0)
    #c0 = np.random.uniform(xmin[0], xmax[0], (self.n_clusters, 1))
    #c1 = np.random.uniform(xmin[1], xmax[1], (self.n_clusters, 1))
    #self.C = np.hstack((c0, c1))
    # Now we need to iterate around the EM algorithm for the number of self.imax
    for _ in range(self.imax):
      # create an empty data matrix
      dist = np.zeros((X.shape[0], self.n_clusters))
      # calculate the distances per center.
      for i in range(self.n_clusters):
        dist[:, i] = self.euchlid(X, self.C[i])
      # assign the data to one of the centroids. Remember we want the minimum distance,
      # between the datapoint and the Centroid.
      X_assign = np.argmin(dist, axis=1)
      # Just in case we want to use the distance metric later let's calculate the
      # total distance of the new assignments to the it's assigned center.
      #dist_metric = np.sum(dist[:, X_assign])
      # Now the final step, let's update the self.C locations. We will use the mean
      # of the assigned points to that cluster.
      for i in range(self.n_clusters):
        self.C[i, :] = np.mean(X[X_assign == i, :], axis=0)

  # Finally let's create a predict method too. This is basically just the distance
  # calculation, and assignment of an input matrix X
  def predict(self, X):
    # create an empty distance matrix
    dist = np.zeros((X.shape[0], self.n_clusters))
    # calculate the distances
    for i in range(self.n_clusters):
      dist[:, i] = self.euchlid(X, self.C[i])
    # return the assignments.
    return np.argmin(dist, axis=1)

# Arranging data
train_data = np.vstack([picture_background["train"], picture_red["train"]])
print(train_data.shape)
# Creating k_means class
kmeans = KMeans(2)

# Fitting data
kmeans.fit(train_data)
Y = kmeans.predict(train_data)
print(Y[Y == 0].sum(), Y[Y == 1].sum())
labels = kmeans.labels_
print(labels)
labels=list(labels)
# now use our data on this
# Let's use this class to cluster some data (D0 from exercise 1) with 4 clusters to start with
# and an imax of zero (we only randomly assign centers).
# Create the object
#kmeans = KMeans(2, imax=100)
#X = picture_background  # I did this so I could easily change later.
#kmeans.fit(X)
#Y = kmeans.predict(X)
# let's plot what this looks like
# but first we want to know the  unique values in D0a so we aren't constantly having
# to change the label values, you'll need np.unique
Yu = np.unique(Y)
print(Yu.shape)
# now scatter plt based on the predictions
plt.figure()
for i in Yu:
  plt.scatter(train_data[Y == i, 0], train_data[Y == i, 1], label='{}'.format(i))
plt.legend()
# plt.show()
plt.close()
# Now what's wrong with what we did? We fit and predicted on the same  data.
# Go back and Fit with D0 and predict with D1... How does that look?
# Now we need to evaluate this,  for that we will use
# from sklearn.metrics import completeness_score as skcs
# Which is a metric designed expressly for clustering.
# You will need to reshape the L vectors to be np.shape = (N,)

#acc = skcs(L0.reshape((-1,)), Y)
#print(acc)

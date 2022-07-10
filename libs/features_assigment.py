"""
  This is a file that will do some basic lbp stuff in a library file.
  Import your libraries directly below this string.
"""
from skimage.feature import local_binary_pattern as lbp
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.cluster import KMeans

def extract_lbp_feature(file,
                        radius=1,  # radius
                        npoints=8,  # the number of points around the radius.
                        nbins=128,  # histogram
                        range_bins=(0, 255)):  # the range
    rgb = file
    gry = rgb2gray(rgb)
    feat = lbp(gry, R = radius, P = npoints)
    feats, edges = np.histogram( feat, bins = nbins, range = range_bins)
    return feat, edges



# Extract the hog information from a file path, coverts to greyscale.
# Input: filname, hog orientation, hog pixel per, hog cells per,
# Output: feature vector
def extract_hog_matrix(f, o, p, c):
  # convert to greyscale
  gry = rgb2gray(f)
  # calculate the HOG representation
  feat = hog(gry, orientations = o,
             pixels_per_cell = p,
             cells_per_block = c,
             visualize = False,
             feature_vector = False)
  # return a feature of shape (-1, orientations)
  return feat.reshape((-1, o)) # m,8, h/p*w/p*c*0

# convert the training dictionary
# the kmeans classifier.

# output will be the full feature vector for kmeans
def extract_full_hog_features(X, o, p, c):
  # iterate over the dictionary
  firstfile = True
  for t, v in X.items():
    for f in v:
      # extract hog from the file
      feat  = extract_hog_matrix(f, o, p, c)
      # concatenate features
      if firstfile:
        fullvec = feat
        firstfile = False
      else:
        fullvec = np.vstack((fullvec, feat))
  # Return full vecetor
  return fullvec


# kmeans based BoVW classifier
# Create class
class BoVW():
  # Initialise with the number of clusters
  def __init__(self, num_clusters):
    self.num_clusters = num_clusters

  # the fit function to fit our kmeans to a feature vector of size (-1, dimensions)
  def fit(self, X):
    # create and fit the kmeans object
    self.kmeans = KMeans(self.num_clusters)
    self.kmeans.fit(X)

  # return a histogram based on the kmeans algorithm plus the number of clusters
  def predict (self, X):
    fv = self.kmeans.predict(X)
    # use np.histogram to get the histogram
    h, _ = np.histogram(fv, bins = self.num_clusters)
    return h

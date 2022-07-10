"""
assignment
"""

import libs as libs
import skimage.color
from skimage.io import imsave, imread, imshow, show, imshow_collection
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import classification_report
from time import time


#### The flags for KMeans ####
part_1 = True
print_1 = True
part_2 = True
visualisation = True
mask_part_1 = True
mask_show = True
mask_part_2 = True
#### The flags for Multivariate Gaussian ####
solution_2 = True
ex03 = True
ex04 = True

# ===== Loading of the data =====
#number one:
print("background")
with open("PAML_data/Q1_BG_dict.pkl", "rb") as sf:
   picture_background = pickle.load(sf)
   #print (picture_background)


with open("PAML_data/Q1_Red_dict.pkl", "rb") as sf:
    picture_red = pickle.load(sf)
    #print (picture_red)

print("yellow")

with open("PAML_data/Q1_Yellow_dict.pkl", "rb") as sf:
    picture_yellow = pickle.load(sf)
    #print(picture_yellow)



#number two:
print("BG")

with open("PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)
    #print(picture_BG)


print("SP")

with open("PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)
    #print(picture_SP.keys())


# the data is divided into 3 sets: 'train', 'validation', 'evaluation'
#The size of the data matrix X for each set is: (10000, 3) (5000, 3) (5000, 3): Many pictures especially in
#training data set, but also each 5000 in the validation and evaluation data sets
#he size of each entry is: (64, 64, 3) (64, 64, 3) (64, 64, 3): These are very big pictures.




for fname in ['PAML_data/Q1_BG_dict.pkl',
              'PAML_data/Q1_Red_dict.pkl',
              'PAML_data/Q1_Yellow_dict.pkl']:
	print("PAML_data/Q1_BG_dict.pkl'", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:",
              data.keys())
		print("The size of the data matrix X for each set is:",
              data['train'].shape,
              data['validation'].shape,
              data['evaluation'].shape)


for fname in ['PAML_data/Q2_BG_dict.pkl',
              'PAML_data/Q2_SP_dict.pkl']:
	print("PAML_data/Q2_BG_dict.pkl", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The number of entries for each set is:",
              len(data['train']), len(data['validation']),
              len(data['evaluation']))
		print("The size of each entry is:",
              data['train'][0].shape,
              data['validation'][0].shape,
              data['evaluation'][0].shape)

# ===== End of loading =====
# ===== Creating k_means classes =====
start =time()


if part_1:


    train_data_merge = np.vstack([picture_background["train"],
                                     picture_red["train"],
                                     picture_yellow["train"]])

    kmeans_every = KMeans(8)
    # Question what the optimal cluster number is. For 6 Clusters yellow/orange gets separated from red the first time.

    # Now the color space hsv will be used
    colourspace = "hsv"

    train_data_together = skimage.color.convert_colorspace(train_data_merge,
                                                           fromspace= "rgb",
                                                           tospace=colourspace)


    # Fit data for the usage of _n_threads in the kmeans object
    kmeans_every.fit(train_data_together)

if print_1:
    print(kmeans_every.cluster_centers_)

if part_2:
    the_red_validation = skimage.color.convert_colorspace(picture_red["validation"],
                                                           fromspace= "rgb",
                                                           tospace=colourspace)
    red_val_data = kmeans_every.predict(the_red_validation)
    the_yellow_validation = skimage.color.convert_colorspace(picture_yellow["validation"],
                                                              fromspace= "rgb",
                                                              tospace=colourspace)
    yellow_val_data = kmeans_every.predict(the_yellow_validation)


    # ===== Classification array =====

    def cluster_appearance(classification_array):
      """ Where is the most red? And i need a list where red is at least one time present"""
      appearance = {}
      for num in np.unique(classification_array):
        # Fill dictionary
        occ = classification_array.tolist().count(num)
        appearance.update({num: occ})
      # Here the class with the most common appearance is selected, for the
      #classifier to select f.e. red in terms of the most common appearance of the color
      max_occ =  max(zip(appearance.values(), appearance.keys()))[1]
      return max_occ.astype(int)

    # Here red has the most common appearance
    The_cluster_red =  cluster_appearance(red_val_data)
    The_cluster_yellow = cluster_appearance(yellow_val_data)
    print("Red cluster: {}\nYellow cluster: {}".format(The_cluster_red, The_cluster_yellow))

    # ===== Classification array End =====

    if visualisation:
        # ===== Visualisation =====
        labels_all = kmeans_every.labels_
        print(labels_all)
        labels = list(labels_all)
        show_the_centroid = kmeans_every.cluster_centers_
        print("centroids")
        print(show_the_centroid)
        centroid_show_1 = skimage.color.convert_colorspace(show_the_centroid,
                                                           fromspace=colourspace,
                                                           tospace="rgb")
        centroid_show_2 = centroid_show_1 * 255
        print(centroid_show_2)

        use_percentage = []
        for i in range(len(centroid_show_2)):
            j = labels.count(i)
            j = j / (len(labels))
            use_percentage.append(j)
        print(use_percentage)

        plt.pie(use_percentage,
                colors=np.array(centroid_show_2 / 255),
                labels=np.arange(len(centroid_show_2)))
        plt.show()
        # ===== Visualisation End =====


# =====Now to make a mask, an image is read in and worked with =====
if mask_part_1:

    for img_input in picture_SP["train"]:
        #first image will be transformed
       picture_reshape = img_input.reshape(-1, img_input[0, 0, :].size)
       picture_converted = skimage.color.convert_colorspace(picture_reshape,
                                                          fromspace="rgb",
                                                          tospace=colourspace)
       print(picture_converted.shape)
       index_classes = kmeans_every.predict(picture_converted)
       the_zero_mask = np.zeros(picture_converted.shape, dtype=int)
       for_the_yellow_pixel, for_the_red_pixel = 0, 0
       for i in range(the_zero_mask.shape[0]):
         if index_classes[i] == The_cluster_yellow:
           the_zero_mask[i] = np.array([255, 255, 0], dtype=int)
           for_the_yellow_pixel += 1
         elif index_classes[i] == The_cluster_red:
           the_zero_mask[i] = np.array([255, 0, 0], dtype=int)
           for_the_red_pixel += 1
         else:
           the_zero_mask[i] = np.array([0, 255, 0], dtype=int)
       msk = the_zero_mask.reshape(img_input.shape)
       print("Red pixel: {:0.01f}%,"
             " Yellow pixel: {:0.01f}%, "
             "Background pixel: {:0.01f}%".format(for_the_red_pixel * 100 / the_zero_mask.shape[0],
                                               for_the_yellow_pixel * 100 / the_zero_mask.shape[0],
(the_zero_mask.shape[0] - for_the_yellow_pixel - for_the_red_pixel) * 100 / the_zero_mask.shape[0]))

if mask_show:

       imshow(msk, vmin=0, vmax=255)
       plt.figure()
       plt.imshow(msk, vmin=0, vmax=255)
       plt.imshow(img_input)
       plt.show()

if mask_part_2:

    # ===== Evaluation labels red and background =====
    bg_red_eva1 = np.vstack([picture_background["evaluation"], picture_red["evaluation"]])
    bg_red_eva2 = skimage.color.convert_colorspace(bg_red_eva1, fromspace = "rgb", tospace = colourspace)
    the_length_of_background = len(picture_background["evaluation"])
    the_length_of_red = len(picture_red["evaluation"])
    the_labels_evaluation_of_both_bg_red = np.concatenate([np.zeros(the_length_of_background,
                                                                    dtype = int),
                                                           np.ones(the_length_of_red, dtype = int)])
    # ===== Evaluation labels red and background End =====

    # ===== Extend Evaluation labels yellow  =====
    the_evaluation_of_all = np.vstack([picture_background["evaluation"], picture_red["evaluation"],
                                       picture_yellow["evaluation"]])
    the_evaluation_of_all = skimage.color.convert_colorspace(the_evaluation_of_all, fromspace = "rgb",
                                                             tospace = colourspace)
    the_length_of_yellow = len(picture_yellow["evaluation"])
    the_evaluation_of_all_labels = np.concatenate([the_labels_evaluation_of_both_bg_red,
                                                   np.full(the_length_of_yellow, 2)])
    # ===== Extend evaluation labels yellow End =====

    # ===== Create labels based on prediction on the evaluation dataset for background and red =====
    prediction_evaluation_of_both_bg_red = kmeans_every.predict(bg_red_eva1)
    tmp = np.zeros(prediction_evaluation_of_both_bg_red.shape, dtype=int)
    for i, cluster in enumerate(prediction_evaluation_of_both_bg_red):
      if cluster == The_cluster_red:
        tmp[i] = 1
    prediction_evaluation_of_both_bg_red = tmp

    the_prediction_evaluation_of_all1 = kmeans_every.predict(the_evaluation_of_all)
    tmp = np.zeros(the_prediction_evaluation_of_all1.shape, dtype=int)
    for i, cluster in enumerate(the_prediction_evaluation_of_all1):
      if cluster == The_cluster_red:
        tmp[i] = 1
      elif cluster == The_cluster_yellow:
        tmp[i] = 2
    the_prediction_evaluation_of_all = tmp
    # ===== Create labels based on prediction on the evaluation dataset for background and red End =====

    # ===== f1score ====
    p, r, t = prc(the_labels_evaluation_of_both_bg_red, prediction_evaluation_of_both_bg_red)
    f1 = 2*p*r/(p+r+0.0000001)
    am = np.argmax( f1 )
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( 'Background and red data Precision Recall: F1-score of {:0.04f}'.format( f1[am] ) )
    plt.show()
    # ===== f1score End====

    # ===== The calculation of the two accuracy scores as well as of the confusion matrices =====
    acc_lin = accuracy_score( the_labels_evaluation_of_both_bg_red, prediction_evaluation_of_both_bg_red )
    print( 'Accuracy of the bg and red data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( the_labels_evaluation_of_both_bg_red, prediction_evaluation_of_both_bg_red ) )


    acc_lin = accuracy_score( the_evaluation_of_all_labels, the_prediction_evaluation_of_all)
    print( 'Accuracy of the bg, red and yellow data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( the_evaluation_of_all_labels, the_prediction_evaluation_of_all) )


    # ===== No 1 End =====


print("solution 2")
if solution_2:


    class MultivariateGaussian:
        # Create the __init__ function, you will also need to initialise the base class: super().__init__()
        # This class will also take two inputs: mu and sigma which default to an empty list each.
        # If both of these members are not empty you should run the _precalculations method
        # which we will code up next.
        def __init__(self, mu=[], sigma=[]):
            super().__init__()
            self.mu = mu
            self.sigma = sigma
            if (not (self.sigma == []) and (not (self.mu == []))):
                self._precalculations()

        # When we perform the log likelihood calculation we need to calculate some values
        # including the Sigma^-1 and |Sigma| as you can see in the pdf. Along with these
        # values we will also precompute the constant values from the pdf.
        # Create a method called _precalculations with no inputs.
        def _precalculations(self):
            # How many dimensions do we have?
            n = self.mu.shape[1]

            # Calculate the inverse matrix using np.linalg.inv and store as a member
            self.inv_sigma = np.linalg.inv(self.sigma)

            # calculate the two constant values from the pdf.
            # the log determinant can be calculated by np.linalg.slogdeg()
            log_two_pi = -n / 2. * np.log(2 * np.pi)
            log_det = -0.5 * np.linalg.slogdet(self.sigma)[1]

            # now sum these two constants together and store them as a member.
            self.constant = log_two_pi + log_det

        # Next we will overwrite the log_likelihood method from the base class.
        def log_likelihood(self, X):
            # get the shape of the data (m samples, n dimensions)
            m, n = X.shape

            # create an empty log likelihood output to the shape of m
            llike = np.zeros((m,))

            # calculate the residuals X - mu
            resids = X - self.mu

            # iterate over the number of data points (m) in residuals and calculate the log likelihood for each.
            # equation in the pdf, using the members created in _precalculations.
            # Hopefully, you see the benefit of precalculating the constants and inverse.
            for i in range(m):
                llike[i] = self.constant - resids[i, :] @ self.inv_sigma @ resids[i, :].T

            # return the log likelihood values
            return llike

        # Now we will overwrite the train function.
        def train(self, X):
            # get the shape of the data
            m, n = X.shape

            # step 1 estimate the mean values. X is of size (m,n) and take the sum over m samples.
            # then divide by the total number of samples.
            mu = np.sum(X, axis=0) / float(m)
            mu = np.reshape(mu, (1, n))

            # Step 2 calculate the covariance matrix
            # residuals
            norm_X = X - mu

            # covariance n,n = (n,m @ m,n) / float( m )
            sigma = (norm_X.T @ norm_X) / float(m)

            # Assign class values and compute internals
            self.mu = mu
            self.sigma = sigma

            # step 3 precalcuate the internals for log likelihood
            self._precalculations()


    colourspace = "hsv"
    train_data_bg = skimage.color.convert_colorspace(picture_background["train"], fromspace="rgb", tospace=colourspace)
    train_data_red = skimage.color.convert_colorspace(picture_red["train"], fromspace="rgb", tospace=colourspace)
    validation_data_bg_red = skimage.color.convert_colorspace(np.vstack([picture_background["validation"], picture_red["validation"]]), fromspace="rgb", tospace=colourspace)

    mvg_bg = MultivariateGaussian()
    mvg_bg.train(train_data_bg)
    mvg_red = MultivariateGaussian()
    mvg_red.train(train_data_red)

    loglike = np.zeros((validation_data_bg_red.shape[0], 2))
    loglike[:, 0] = mvg_bg.log_likelihood(validation_data_bg_red)
    loglike[:, 1] = mvg_red.log_likelihood(validation_data_bg_red)
    print("log")

    classified = np.argmax(loglike, axis=1)

    evaluation_data_bg_red = np.vstack([picture_background["validation"], picture_red["validation"]])
    evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace="rgb",
                                                              tospace=colourspace)
    the_length_of_background = len(picture_background["validation"])
    the_length_of_red = len(picture_red["validation"])
    eval_labels_bg_red = np.concatenate([np.zeros(the_length_of_background, dtype=int), np.ones(the_length_of_red, dtype=int)])

    acc = accuracy_score(eval_labels_bg_red, classified)
    print('Accuracy of the MVGs is:', acc)
    print( confusion_matrix( eval_labels_bg_red, classified ) )

    # now the f1score stuff.
    p, r, t = prc(eval_labels_bg_red, classified)
    # print( 't', len( t ) )
    f1 = 2*p*r/(p+r+0.0000001)
    am = np.argmax( f1 )
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( 'Background and red data Precision Recall: F1-score of {}'.format( f1[am] ) )
    plt.show()

    print(classification_report(eval_labels_bg_red, classified.reshape( (-1,1) ) ))



    acc = accuracy_score( eval_labels_bg_red, classified.reshape( (-1,1) ) )
    print( 'Accuracy of the GMM based prediction is:', acc )
    sc = loglike[:,0].reshape( (-1,1) )
    numthr = 200
    f1val = np.zeros( ( numthr, ) )
    precision = np.zeros( ( numthr, ) )
    recall = np.zeros( ( numthr, ) )
    thrs = np.linspace( sc.min(), sc.max(), numthr )
    bestf1 = 0
    bestth = 0
    for i, th in enumerate( thrs ):
        o = classification_report(eval_labels_bg_red, sc <= th, output_dict=True, zero_division=0)
        f1val[i] = o["1"]['f1-score']
        precision[i] = o["1"]['precision']
        recall[i] = o["1"]['recall']
        if f1val[i] > bestf1:
            bestf1 = f1val[i]
            bestth = th



    print(bestth)
    # -8.985876310917995 threshold
    #Accuracy: 0.9284 with optimisation of threshold
    #With this threshold the model is working best
    print(bestf1)
    print(classification_report(eval_labels_bg_red, sc <= bestth, zero_division=0))
    #log likelihood propably score to not iterate over. Use <=



import pickle


pickle_out = open("mvg_bg.pickle","wb")
pickle.dump(mvg_bg, pickle_out)
pickle_out.close()


import pickle

pickle_out = open("kmeans_every","wb")
pickle.dump(kmeans_every, pickle_out)
pickle_out.close()

end = time()
dtime = end - start
print(dtime)
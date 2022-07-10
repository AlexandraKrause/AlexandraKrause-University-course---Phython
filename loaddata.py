"""
  This loads the data for the assignment
"""

import pickle

"""
  Question 1 example
"""
# load the background data
with open( 'Q1_BG_dict.pkl', 'rb' ) as fid:
  data = pickle.load( fid )
print( type( data ) )
# >> <class 'dict'>
print( data.keys() )
# >> dict_keys(['train', 'validation', 'evaluation'])
train = data['train']
valid = data['validation']
eval = data['evaluation']
print( type( train ) ) # all three are the same.
# >> <class 'numpy.ndarray'>
print( train.shape )
# >> (100000, 3)
print( valid.shape )
# >> (50000, 3)
print( eval.shape )
# >> (50000, 3)

"""
  Question 2 example
"""
# load the sweet pepper data
with open( 'Q2_SP_dict.pkl', 'rb' ) as fid:
  data = pickle.load( fid )
print( type( data ) )
# >> <class 'dict'>
print( data.keys() )
# >> dict_keys(['train', 'validation', 'evaluation'])
train = data['train']
valid = data['validation']
eval = data['evaluation']
print( type( train ) ) # all three are the same.
# >> <class 'list'>
print( len( train ) )
# >> 100
print( type( train[0] ) )
# >> <class 'numpy.ndarray'>
print( train[0].shape ) # all images in all subsets are the same size!
# >> (64, 64, 3)

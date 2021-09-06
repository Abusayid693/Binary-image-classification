from sklearn import utils

def prepareDataSets(positive_set,negative_set):
    
    """
    Args:
    positive_set: Positive data-sets (i.e datasets with label 1)
    negative_set: Negative data-sets (i.e datasets with label 0)

    Returns:
     X: Training sets (both 1 and 0)
     Y: Corressponding labels for training sets
    """
    
    # Y labels for datasets
    y_pos=np.ones((positive_set.shape[1],1))
    y_neg=np.zeros((negative_set.shape[1],1))
    
    # Concatenate all seperate datasets to from main universerl sets
    Y=np.concatenate((y_pos, y_neg),axis=0)
    X=np.concatenate((positive_set, negative_set),axis=1)
   
    """
    Random shuffling :
       While shuffling X and Y randomly make sure they 
       done together to avoid wrong shuffling for labels
       and training sets.
    """
    rng_state = np.random.get_state()
    np.random.shuffle(Y)
    np.random.set_state(rng_state)
    np.random.shuffle(X.T)
    
    return X,Y.T



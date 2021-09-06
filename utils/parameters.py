
def parameter_initialization(v):
    
    """
    The function initialises parameters for model training
    
    args : 
      v -- Number of features in the training set (X.shape[1])
    
    Returns :
     w -- initialized vector 
     b -- initialized vector
    """
    
    
    w=np.zeros((v,1))
    b=0.0
    return w,b
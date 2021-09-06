def forward_propogation(X,Y,w,b):    
    """
    Args : 
    
    X - Training data-sets
    Y - Corresponding labels
    w - parameter
    b - parameter
    
    Returns :
    
    grads : Overall gradient 
    const : Overall cost
    
    """
    
    """
    Getting number of training sets 
    Note : In this case the training sets in of shape : 
           [No_of_features,no_of_trainig_examples] 
    """ 
    m=X.shape[1]
    
    # Predictive model calculation
    z=np.dot(w.T,X)+b
    a=sigmoid(z) 
    # Backpropogation
    dz=a-Y 
    dw=(1/m)*np.dot(X,dz.T)
    db=(1/m)*np.sum(dz) 
    temp_1 = np.log(a)
    temp_2 = np.log(1-a)
    
    # Loss function on overall datasets
    cost=-(np.dot(Y,temp_1.T))-np.dot(1-Y,temp_2.T)
    cost=(1/m)*cost
    
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):   
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)  
    costs=[]  
    for i in range(num_iterations):        
        grads,cost=forward_propogation(X,Y,w,b)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w=w-(learning_rate*dw)
        b=b-(learning_rate*db)
                
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)        
            # Print the cost every 100 training iterations
            if print_cost:                
                print ("Cost after iteration %i: %f" %(i, cost))
            
    prams={"w":w,
               "b":b}    
    grads={"dw":dw,
              "db":db}    
    return prams,grads,costs    


def predict(X,w,b):  
    m=X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    z=np.dot(w.T,X)+b
    a=sigmoid(z)
    
    for i in range(m):
        if a[0,i]>=0.5:
            Y_prediction[0,i]=1
         
        else:
            Y_prediction[0,i]=0      
    return Y_prediction


 def model_train(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate,print_cost): 
        
    w,b=initialize_with_zeros(X_train.shape[0])
    prams,grads,costs=optimize(w, b, X_train, Y_train, num_iterations, learning_rate,print_cost)
    
    w = prams["w"]
    b = prams["b"]
    
    Y_prediction_train= predict(X_train,w,b)
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        
    d = {"costs": costs,
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}    
    return d   

 logistic_regression_model = model(X, Y, [], [], 2000, 0.005, True)   
costs=logistic_regression_model["costs"]
plt.plot(costs)

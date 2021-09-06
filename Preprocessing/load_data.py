def loadDatasets(type_of,num):
    
    """ Loading datasets from local files
    
        Args:
        - type_of : Positive/Negative datasets
        - num : Number of datasets per group
    """
    
    image=[]
    train_set_arr=[]
    
    for i in range(num):   
        
        """ Image file path
        """
        img="./"+str(type_of)+"/test"+str(i)+".jpg"
        image.append(np.array(Image.open(img).resize((150, 150))))
        train_set_arr.append(np.array(Image.open(img).resize((150, 150))).reshape(-1,1))
        
    return image,train_set_arr 
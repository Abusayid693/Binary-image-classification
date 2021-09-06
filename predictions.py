
def new_image_predictiions(my_image):
   image = np.array(Image.open(my_image).resize((150, 150)))
   plt.imshow(image)
   image = image / 255.
   image = image.reshape((1, 150 * 150 * 3)).T
   my_predicted_image = predict(image,logistic_regression_model["w"], logistic_regression_model["b"])
   print(my_predicted_image)
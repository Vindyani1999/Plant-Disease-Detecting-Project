# Plant Disease Prediction

## Overview

In this project we are going to build a plant disease prediction system and we are going to use convolutional nural network for the image classification.

## Workflow

### Step 1: Dataset Acquisition
![Dataset Acquisition](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/f678ba27-8129-4805-b3d7-5e1eb2d10df4)

The dataset is from Kaggle. We will get a Kaggle JSON file containing our Kaggle account credentials and retrieve the dataset through an API. (We are going to work with Google Colab)

### Step 2: Data Processing
![Data Processing](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/1af6f2b7-d6c4-48df-9fdc-3e07a1bc3732)

In this data processing step, we will use an image data generator class in TensorFlow. The images are loaded from the directory and prepared to be suitable as input for the neural network.

### Step 3: Data Splitting
![Data Splitting](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/9118cd50-aeee-453d-9342-a8a047edc236)

In this step, we will split our dataset into training and testing data. This step is connected with the data processing because we are going to build a pipeline for the training data generator and validation generator.

### Step 4: Model Building
![Model Building](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/3a9e8c94-035f-476c-b5a6-82b11f11e445)

Next, we will build our convolutional neural network with appropriate convolutional layers and all the required dense layers.

### Step 5: Model Evaluation and Saving
![Model Evaluation and Saving](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/d7f3cd68-1c9e-4264-be4e-7253ba07e30a)

The model is then evaluated to check if it is giving correct predictions and working as expected. After verifying, we can save the model as a file.

### Step 6: Model Export to Streamlit
![Model Export to Streamlit](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/7552ee8f-a19b-4af1-b3af-bcd16356e058)

Using the saved model, we will export this trained model and load it into the Streamlit app's Python script.

### Step 7: Dockerizing the Application
![Dockerizing the Application](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/f1bde005-2306-413d-b564-dd6c5c68b358)

As the first step of Dockerizing, we are going to create a Dockerfile that includes a list of instructions needed to create a Docker image.

### Step 8: Creating Docker Image and Container
![Creating Docker Image and Container](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/56d2f350-5d94-41db-a188-ff218cb4e2b0)

Using this Dockerfile, we can create the Docker image and then create the Docker container.


## Steps 
- firstly, we are going to setup our colab to GPU environment bcz, the model will be train within few time period.
- After creating Kaggle account we can download kaggle.json file to get the credential (settings > API > create new tocken)
- That downloaded folder should be uploaded to directory where we are going to make our model <br>
![image](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/ba122374-960d-46f7-ba4f-a69ee5fff263&width=100&height=120) <br>
- We need to import libraries (colab has already installed those libraries we are going to use)
  > random
  > numpy
  > tensorflow
- Normally when we are training model using CNN there is some randomeness, time to time the result can be somewhat varying. To avoid that we can use random library. In simply it is saying model to start from initially.
- Let say  we are not using that random library and we are not seed the dataset. if there are two people is running same things under same facilities will be have different results. using random library and seeding we can avoid that randomness.
  ```bash
  import random
   random.seed(0)

   import numpy as np
   np.random.seed(0)

   import tensorflow as tf
   tf.random.set_seed(0)
  
- next we have import needed dependencies.
  ```bash
     import os    //Accsess some files
     import json    //To load our kaggle.json file into the file
     from zipfile import ZipFile    //To extract our kaggle dataset
     from PIL import Image   //Pillow library to do some image processing 

     import numpy as np
     import matplotlib.pyplot as plt
     import matplotlib.image as mpimg    //Load and display some images
     from tensorflow.keras.preprocessing.image import ImageDataGenerator    //The data pipline 
     from tensorflow.keras import layers, models    //layers-(convolutional layer, flattening, dense), model-(sequential)

- If you are not using google colab, mannually install kaggle
  ```bash
  !pip install kaggle
- Next, we need to set kaggle username and key as environmental variables as more standard way.
  ```bash
   kaggle_credentails = json.load(open("kaggle.json"))
   os.environ['KAGGLE_USERNAME'] = kaggle_credentails["username"]
   os.environ['KAGGLE_KEY'] = kaggle_credentails["key"]
- next we have to copy the api command from keggle website
![image](https://github.com/Vindyani1999/Plant-Disease-Detecting-Project/assets/145743416/464b37e4-413d-4ce8-a8c0-eb7bebfea036)
   ```bash
   !kaggle datasets download -d abdallahalidev/plantvillage-dataset
- Then the dataset will be downloaded as a zip file. it should be extracted using this code
  ```bash
     
   with ZipFile("plantvillage-dataset.zip",'r') as zip_ref:
   zip_ref.extractall()
- Next we have analyse the dataset such as what are the main catogories what are the diseases types are available etc. When you are expand the directory you can see there are three types of images segmented (without background), color and grayscale. If we think little bit when a user upload an image to identify the dieses most of the time it is colored and it is with background, so tht this analys we are going to select color catogort. (All the catogories has same data)
   ```bash
     
   print(os.listdir("plantvillage dataset"))


   print(len(os.listdir("plantvillage dataset/segmented")))
   print(os.listdir("plantvillage dataset/segmented")[:5])

   print(len(os.listdir("plantvillage dataset/color")))
   print(os.listdir("plantvillage dataset/color")[:5])

   print(len(os.listdir("plantvillage dataset/grayscale")))
   print(os.listdir("plantvillage dataset/grayscale")[:5])

- Now we have selected directory is asign to base directory for fututre purpose
  
   ```bash
   base_dir = 'plantvillage dataset/color'
- Next we are going to plot a selected image from the dataset, if you see, the image size is 256,256
  ```bash
   img = mpimg.imread(image_path)

   print(img.shape)

   # Display the image
   plt.imshow(img)
   plt.axis('off')  # Turn off axis numbers
   plt.show()
  
- 256 size si somewhat big to analyze images, as standard the image size is reduce to 224
  ```bash
   # Image Parameters
   img_size = 224
   batch_size = 32

- When you print and image without using matplotlib or any other library, you can see that it will print multidimensional array with values 1-255. Basically it is not suitable to train a model using nural network, so the we are going to do a scaling such as those all the valies will be in the smae scale.
  
- Why this scaling is needed because, some pixel values are like 20, 30 and some values are like 200,210. at that time the model will not give a smooth performance. Then the gradient of the loss function with respect to weights will not be smooth, that means it not converge correctly, it cant find global minima and it can be stuck into local minima.

  ```bash
   # Image Data Generators
   data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
   )

- Rather than using train test split with for loop that batch method is prefered for deep learning projects because DL projects have more data compare to ML project. In the train split set waht we have doing is by using for loop we are loading full dataset into our memeory. it is not good for large dataset. so taking batch vise is suitable then less number of data is using in the momory.

- In above code we are making a pipline, usinf ImageGenerator class from the tensorflow is helping for that. It will separate 80% data for training and 20% data for testing.

- If u need to increase your dataset what you have do is data augmnetation.

- Now we are going to make the test generator and validation generator

  ```bash
     train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
   )
- Here the intersting part is we dont need to lable our data. Instead of that flow_from_directory can automatically identify the images according to its folder and name.

- In simply we dont need to introduce this is my y and all these are my x values. we dont need to do those things. This is an efficent way to train model in pipline.

- Here we are having the base directory which having path for the dataset.

- then we having target size. we are targeting 224,224 size images. if you have another size of images those all are rescaled in here.

- batch size is another kind of intersting thing. let say if you have a larger ram or memory, you can use higher numbers as batch size. batching is having some advantages. let say we are using first batch for neural network, then according to those data it will adjust their parameters.

- If we are mentioning subset is training it is seeing to data reserves for training (80%).

- same thing is done for validation generator also

  ```bash
  validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
   )

- here we are not using pretrained model to train the model. rather than it we are going to build a simple model

  ```bash
  # Model Definition
  model = models.Sequential()

   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
   model.add(layers.MaxPooling2D(2, 2))

  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D(2, 2))


   model.add(layers.Flatten())
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

- here model.sequential() says that our model is a linear model. That means output of first laye is the input of second layer and so on. sequential() is a mostly used keras API. (keras is a library working on top of tensorflow framework)

- model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3))) in this line  we are adding our layers. it is a 2D layer having 32 which is the number of filters/kernels and (3,3) size of kernel and activation is relu here.

-  model.add(layers.MaxPooling2D(2, 2)) in here we are adding maxpooling layer, expecting to finding different features of the images like edges, spots etc

-  at the end we have flatening layer becuse the first layers can be 2 or 3 dimentional arrays, but in latter layer(dense) we have to convert those 2D or 3D array into 1D array. that is known as flattening.

-  after flatenning we can make dense layers with number of nurones (like 256, 128,...)

- Dense layer is a type of layers which having fully connected neurons.

-  Finally we have to make our output layer which predict the class of the input image. normally there should be neurons equals to classes which we have to classify. for example if we have digit classification project we should have 10 classes in por output layer.

-  each nuron is connected to the output layer is giving some probability. but the nuron which gives the highest probability is getting as the resulted class.

-  After creating the network or model we need to compile it.

     ```bash
      # Compile the Model
      model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

- Then we have to train the model using created nural network
   ```bash
   # Training the Model
   history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Number of steps per epoch
    epochs=7,  # Number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Validation steps
   )
- why we are storing tose every loss and accuracy valuse in the histoey parameter is , then we can draw a grapgh or we can analyse how the loss and accuracy varying in each epoch.

- When you have model.fit() that means you are going to train your model.

- rather than providing training dataset directly, we are suppling or training piplinw which is train generator.

- epoch is let say we have 1000 samples. the model is gone through all those 1000 samples for one time that means one epoch.

- here we are using batch wise inputs so we have to define steps for epoch, that means how much model have to train for fill one epoch. it is get by all samples in train set should devide by batch size.

- validation is providing as the same way

- After doing model eveluation, we need to go to the building the predictive system

  ```bash
  // Function to Load and Preprocess the Image using Pillow
   def load_and_preprocess_image(image_path, target_size=(224, 224)):
  
       // Load the image
       img = Image.open(image_path)
  
       // Resize the image
       img = img.resize(target_size)
  
       // Convert the image to a numpy array
       img_array = np.array(img)
  
       // Add batch dimension
       img_array = np.expand_dims(img_array, axis=0)
  
       // Scale the image values to [0, 1]
       img_array = img_array.astype('float32') / 255.
       return img_array
   
   // Function to Predict the Class of an Image
   def predict_image_class(model, image_path, class_indices):
       preprocessed_img = load_and_preprocess_image(image_path)
       predictions = model.predict(preprocessed_img)
       predicted_class_index = np.argmax(predictions, axis=1)[0]
       predicted_class_name = class_indices[predicted_class_index]
       return predicted_class_name
  ```
  - Here we have creted 2 functions.
  - > First one for image loading
  - > Second one for class predicting
 
  - In the first function we have image path for taking the image and target size, in the actual senario we dont know what size image is going to provide as input by the user. so we have to give target size to resize it.
  - We are going to open that uploaded image using PIL library
  - and resizing it using target sizr (224,224)
  - after that we have to convert those image into numpy array to do prediction
  - we ahve to resize the all pixels in the array by dividing 255
  - finally return the rescaled image
 
  - Now the image is suitable to passes through the network and do a prediction
 
  - second function has three parametes model which we are going to use to have a prediction, image path which is the path uploaded image is presented and class_indices. what is this class_indices. we know that after done the prediction the output layer is saying index 16th index is having maximum probability like that. but user doesnt know what is this 10th index or 16index. so we have to create a dictionary which include indexes and corresponded class name.
    ```bash
    class_indices = {v: k for k, v in train_generator.class_indices.items()}

    ```
  - prediction is done by predictions = model.predict(preprocessed_img) using scaled image
  - this prediction gives 38 probability values becuse we are having 38 classes. now we need to get the class index of which having maximum probability using predicted_class_index = np.argmax(predictions, axis=1)[0]
 
  - Now we are going to save our class indexes in json format and then we can download it and use in our streamlit application.
 
    ```bash
    json.dump(class_indices, open('class_indices.json', 'w'))
    ```
  - For saving the model you can mount your google drive or directly save into path which u are working on.
  ```bash
  model.save('plant_disease_prediction_model.h5')
  ```
  

  

# CS6910 Assignment 2 Part A

The goal of this part is to train a custom CNN model for image classification on the iNaturalist dataset from scratch. The wandb report for the assignment can be found [here](https://wandb.ai/hithesh-sidhesh/Assignment_2/reports/CS6910-Assignment-2--VmlldzoxNzI3Nzcy).

## NOTE:

1. The Part_A.ipynb file contains the entire code used for the wandb report metrics and plots, along with the output cells for the same.
2. The partA.py file contains the functions and code for running a model using user parameters passed by command line arguments and to generate the model and plots corresponding to the test data for the same.


## Training the model

To create the model, the following parameters are required:


      1. number_of_conv_layers:Number of convolution layers in the model
      2. number_of_filters:Number of filters to use per layer
      3. kernel_size:Size of the kernel for the model
      4. image_shape:Shape of the image
      5. activation_conv:Activation for convolution layer
      6. pool_size:Size of maxpool layer
      7. neurons_dense:Neurons in the dense layer
      8. activation_dense:Activation for dense layer
      9. neurons_output:Number of neurons in the output layer 
      10. activation_output:Output layer activation
      11. epochs:Number of epochs to run the model (default 2)
      12. batch_size: Batch size (default 32)
      13. augment_data:Augment data or not (default False)
      14. dropout_rate Dropout rate (default 0.1)
      15. batch_norm:Apply batch normalization or not (default False)
      16. log: Log onto wandb or not (default True)

The function which takes these parameters, prepares the train and validation data accordingly, runs the model and returns the model can be called as follows:

```python

number_of_conv_layers=5

number_of_filters = [16, 32, 64, 128, 256]

kernel_size=[(3,3)]*number_of_conv_layers

batch_norm = True

image_shape=(128,128,3)

activation_conv=['relu','relu','relu','relu','relu']

pool_size=[(2,2)]*number_of_conv_layers

neurons_dense_layer=[64, 128]

activation_dense='relu'

no_of_output_classes=10

activation_output='softmax'

epochs=21

batch_size = 32

dropout_rate = 0.4

augment_data = True

model = build_model(number_of_conv_layers,number_of_filters,kernel_size,image_shape,activation_conv,pool_size,neurons_dense_layer,activation_dense,no_of_output_classes,activation_output, epochs=epochs, batch_size = batch_size, augment_data = augment_data, dropout_rate = dropout_rate, batch_norm=batch_norm, log=False)
```

## Passing parameters using command line argument

To train the model using parameters passed by command line argument, the following process needs to be followed:

      1. Use the partA.py file for this process.
      2. Update the filePath variable with the appropriate path for the dataset.
      3. Run the python file and enter the arguments when prompt appears on the command line.
      
The following parameters need to be passed using the command line:

      1. number_of_conv_layers:Number of convolution layers in the model (integer)
      2. number_of_filters:Number of filters to use per layer(integer)
      3. kernel_size:Size of the kernel for the model (integer, eg:2 if kernel size is (2,2))
      4. batch_norm:Apply batch normalization or not ("yes" or "no")
      5. pool_size:Size of maxpool layer (integer, eg:3 if (3,3) s maxpool size)
      6. neurons_dense:Neurons in the dense layer (list of integer, eg: [64,128])
      7. epochs:Number of epochs to run the model (integer)
      8. batch_size: Batch size (integer)
      9. augment_data:Augment data or not ("yes" or "no")
      10. dropout_rate Dropout rate (float, eg:0.1)
      
## Testing the trained model on test data

The data preparation part for the test data is as follows:

```python
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
        filePath+'/val',
        target_size=(128,128),
            batch_size=32,
            class_mode='categorical',
            shuffle = False,
        seed = 137)
test_generator = tf.data.Dataset.from_generator(
    lambda: test_gen,
    output_types = (tf.float32, tf.float32)
    ,output_shapes = ([None, 128, 128, 3], [None, 10]),
)
test_generator = test_generator.repeat()
test_generator = test_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```
The test data is then used on the trained model to find the test accuracy. The code snippet for the same is as follows:

```python
  test_loss, test_acc = model.evaluate(test_generator, steps=test_gen.samples//test_gen.batch_size, verbose=2)

```
      
## Architecture of the best model:

The best model is saved in the file "best_model_cnn.h5" and the same can be loaded into the code using the code:
```python
model = load_model(MODEL_PATH)
```

The best model after training and wandb sweeps has the following architecture:

![Model Architecture](https://github.com/HegdeSiddesh/cs6910_Assignment2/blob/main/Part%20A/model_arch.png)




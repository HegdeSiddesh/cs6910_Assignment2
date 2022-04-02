# CS6910 Assignment 2 Part B

The goal of this part is to use a pretrained model and fine tune it for image classification on the iNaturalist dataset from scratch. The wandb report for the assignment can be found [here](https://wandb.ai/hithesh-sidhesh/Assignment_2/reports/CS6910-Assignment-2--VmlldzoxNzI3Nzcy).

## NOTE:

1. The Part_B.ipynb file contains the entire code used for the wandb report metrics and plots, along with the output cells for the same.
2. The partB.py file contains the functions and code for running a model using user parameters passed by command line arguments and to generate the model and plots corresponding to the test data for the same.

## Training the model

To create the model, the following parameters are required:

        1. layers_unfreeze:Number of layers from end of pretrained model to be used while training (default 20)
        2. model:The pretrained model to be used (default "InceptionResNetV2")
        3. dense_layers:Neurons in the dense layer(default [128])
        4. epochs:Number of epochs to tun the model (default 10)
        5. batch_size: Batch size (default 256)
        6. augment_data:Augment data or not (default 'no')
        7. dropout: Dropout rate (default 0.2)
        
The first step is to create an object of ConfigValues() class, which takes these parameters above as input. The code sample for this is as follows:

```python
parameters = ConfigValues(augment_data='no', batch_size=256, dense_layers=[128], dropout=0.2, epochs=5, layers_unfreeze=20, model="InceptionResNetV2")
```

The next step is to call the ``` executePretrainedModel``` function, which takes the ConfigValues type object as a parameter and performs the train and validation data preparation and the model generation and fitting. At the end the trained model is returned by the function.

The code sample for the same is as follows:

```python
model = executePretrainedModel(parameters, log = False)
```
## Testing the trained model on test data

The data preparation part for the test data is as follows:

```python
  test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
  test_gen = test_datagen.flow_from_directory(
          '/content/inaturalist_12K/val',
          target_size=(256,256),
              batch_size=parameters.batch_size,
              class_mode='categorical',
              shuffle = False,
          seed = 137)
  test_generator = tf.data.Dataset.from_generator(
      lambda: test_gen,
      output_types = (tf.float32, tf.float32)
      ,output_shapes = ([None, 256, 256, 3], [None, 10]),
  )
  test_generator = test_generator.repeat()
  test_generator = test_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```
The test data is then used on the trained model to find the test accuracy. The code snippet for the same is as follows:

```python
  test_loss, test_acc = model.evaluate(test_generator, steps=test_gen.samples//test_gen.batch_size, verbose=2)

```

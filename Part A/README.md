# CS6910 Assignment 2 Part A

The goal of this part is to train a custom CNN model for image classification on the iNaturalist dataset from scratch. The wandb report for the assignment can be found [here](https://wandb.ai/hithesh-sidhesh/Assignment_2/reports/CS6910-Assignment-2--VmlldzoxNzI3Nzcy).


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
      11. epochs:Number of epochs to tun the model (default 2)
      12. batch_size: Batch size (default 32)
      13. augment_data:Augment data or not (default False)
      14. dropout_rate Dropout rate (default 0.1)
      15. batch_norm:Apply batch normalization or not (default False)
      16. log: Log onto wandb or not (default True)

The function which takes these parameters, prepares the train and validation data accordingly, runs the model and returns the model can be called as follows:

```python
model = build_model(number_of_conv_layers,number_of_filters,kernel_size,image_shape,activation_conv,pool_size,neurons_dense_layer,activation_dense,no_of_output_classes,activation_output, epochs=epochs, batch_size = batch_size, augment_data = augment_data, dropout_rate = dropout_rate, batch_norm=batch_norm, log=False)
```

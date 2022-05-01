from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU


def convolution(image, filters=64):
    """
    Implements Conv2D -> Batch Normalization -> Conv2D block.
    Args:
        image (array): and image array
        filters (int): Number of output filters in convolution.

    Returns:
        block activation
    """
    convolution = Conv2D(filters, kernel_size = (3,3), padding = "same")(image)
    batch_normalization = BatchNormalization()(convolution)
    activation = ReLU()(batch_normalization)
    
    # Taking first input and implementing the second conv block
    convolution = Conv2D(filters, kernel_size = (3,3), padding = "same")(activation)
    batch_normalization = BatchNormalization()(convolution)
    activation = ReLU()(batch_normalization)
    
    return activation


def encoder(input, filters=64):
    """
    Implements an encoder block.
    Args:
        image (array): and image array
        filters (int): Number of output filters in convolution.

    Returns:
        Encoder output, max pool output
    """

    encoding = convolution(input, filters)
    max_pool = MaxPooling2D(strides = (2,2))(encoding)

    return encoding, max_pool


def decoder(array, skip_input, filters=64):
    """
    Implementes a decoder block
    Args:
        array (array): Upsample from previous layer.
        skip_input (array): inpout from encoder-side skip layer
        filters (int): Number of output filters in convolution.

    Returns:
        Decoded output
    """

    upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(array)
    connect_skip = Concatenate()([upsample, skip_input])
    out = convolution(connect_skip, filters)

    return out

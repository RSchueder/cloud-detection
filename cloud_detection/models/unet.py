from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

from cloud_detection.config import NUM_CLASSES
from cloud_detection.data.generator import data_augmentation
from cloud_detection.models.components import encoder, decoder, convolution


def UNet(image_size):
    image = Input(shape=image_size)
    #x = data_augmentation(image)

    # encoding sequence
    skip1, encoder_1 = encoder(image, 64)
    skip2, encoder_2 = encoder(encoder_1, 64*2)
    skip3, encoder_3 = encoder(encoder_2, 64*4)
    skip4, encoder_4 = encoder(encoder_3, 64*8)
    
    # Preparing the next block
    conv_block = convolution(encoder_4, 64*16)
    
    # Construct the decoder blocks
    decoder_1 = decoder(conv_block, skip4, 64*8)
    decoder_2 = decoder(decoder_1, skip3, 64*4)
    decoder_3 = decoder(decoder_2, skip2, 64*2)
    decoder_4 = decoder(decoder_3, skip1, 64)
    
    classification = Conv2D(NUM_CLASSES, 1, padding="same", activation="sigmoid")(decoder_4)

    model = Model(image, classification)
    
    return model
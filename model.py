from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import load_model
import segmentation_models as sm

pre = sm.Unet(backbone_name='densenet121' , input_shape=(64,64,3))

# Pyramid feature extraction
def model():
  C = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(pre.input)
  # I = Input(shape=(32,32,3))
  c1= Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(pre.layers[52].input)
  c2 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c1)
  c3 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c2)
  # B1 = BatchNormalization()(c2)
  M1= AveragePooling2D((2,2))(c3)

  c4= Conv2D(64 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(M1)
  c5 = Conv2D(64 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c4)
  c6 = Conv2D(64 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c5)
  # B1 = BatchNormalization()(c2)
  M2= AveragePooling2D((2,2))(c6)



  c7= Conv2D(128 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(M2)
  c8 = Conv2D(128 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c7)
  # c5 = BatchNormalization()(c5)
  c9 = Conv2D(128, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c8)
  c10 = Conv2D(128 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c9)
  # B2 = BatchNormalization()(c7)
  M3= AveragePooling2D((2,2))(c10)


  c11= Conv2D(256 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(M3)
  c12 = Conv2D(256 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c11)
  # c9 = BatchNormalization()(c9)
  c13 = Conv2D(256, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c12)
  c14 = Conv2D(256 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c13)
  c15 = Conv2D(256 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(c14)
  # B3 = BatchNormalization()(c12)
  M4 = AveragePooling2D((2,2))(c15)


  # //////////////

  # Expansion
  up1 =UpSampling2D((2,2))(M4)
  merge_1 = concatenate([c15 ,up1] , axis =3)
  conv_1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1)
  conv_1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)
  # conv_1 = BatchNormalization()(conv_1)
  conv_1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)
  conv_1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)
  conv_1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)
  # bn_1 = BatchNormalization()(conv_1)


  up2 = UpSampling2D((2,2))(conv_1)
  merge_2 = concatenate([c10 , up2 ] , axis =3)
  conv_2 = Conv2D(128, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(merge_2)
  conv_2 = Conv2D(128, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_2)
  # conv_2 = BatchNormalization()(conv_2)
  conv_2 = Conv2D(128, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_2)
  conv_2 = Conv2D(128, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_2)
  # bn_2 = BatchNormalization()(conv_2)


  up3 = UpSampling2D((2,2))(conv_2)
  merge_3 = concatenate([c6 , up3 ] , axis =3)
  conv_3 = Conv2D(64 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(merge_3)
  conv_3 = Conv2D(64 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_3)
  # conv_3 = BatchNormalization()(conv_3)
  conv_3 = Conv2D(64, (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_3)
  # bn_3 = BatchNormalization()(conv_3)


  up4 = UpSampling2D((2,2))(conv_3)
  merge_4 = concatenate([pre.layers[7].output , up4 ] , axis =3)
  conv_4 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(merge_4)
  conv_4 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_4)
  conv_4 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_4)


  up5 = UpSampling2D((2,2))(conv_4)
  merge_5 = concatenate([pre.layers[2].output , up5 ] , axis =3)
  conv_5 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(merge_5)
  conv_5 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_5)
  conv_5 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(conv_5)

  up6 = UpSampling2D((2,2))(conv_5)
  merge_6 = concatenate([C , up6 ] , axis =3)
  conv_6 = Conv2D(32 , (3,3) , activation='relu' ,kernel_initializer='he_normal' , padding='same')(merge_6)

  # conv = Conv2D(2, (1,1), activation = 'relu' ,kernel_initializer='he_normal', padding ='same')(conv_3)

  conv_last = Conv2D(1, 1, activation = 'sigmoid')(conv_6)
  # /////////////////////////////
  model = Model(pre.input , conv_last)
  # print(model.summary() , len(model.layers))
  return model

model = model()
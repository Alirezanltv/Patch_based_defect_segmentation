def extract_patches(images):
  # images mean X,Y
  ksize_rows = 64
  ksize_cols = 64

  # strides_rows and strides_cols determine the distance between
  #+ the centers of two consecutive patches.
  strides_rows = 64 # 128
  strides_cols = 64 # 128

  # The size of sliding window
  ksizes = [1, ksize_rows, ksize_cols, 1]

  # How far the centers of 2 consecutive patches are in the image
  strides = [1, strides_rows, strides_cols, 1]

  rates = [1, 1, 1, 1] # sample pixel consecutively

  padding='VALID' # or 'SAME'

  sess = tf.compat.v1.Session()
      
      
  image_patches = tf.image.extract_patches(X, ksizes, strides, rates, padding)
  print(image_patches.shape)


  label_patches = tf.image.extract_patches(Y, ksizes, strides, rates, padding)
  print(label_patches.shape)


  image_patches_val= tf.image.extract_patches(X_val, ksizes, strides, rates, padding)
  label_patches_val= tf.image.extract_patches(Y_val, ksizes, strides, rates, padding)
  print(image_patches_val.shape , label_patches_val.shape)



  num_patches = image_patches.shape[1]*image_patches.shape[2]
  h_patch = ksize_rows

  print(num_patches , h_patch)

  I = tf.reshape(image_patches , (len(X)*num_patches,h_patch,h_patch,3))
  L = tf.reshape(label_patches , (len(X)*num_patches,h_patch,h_patch,1))


  total_image_patches_train = tf.reshape(I ,(len(X)*num_patches,h_patch,h_patch,3))
  total_label_patches_train = tf.reshape(L ,(len(X)*num_patches,h_patch,h_patch,1))


  total_image_patches_val = tf.reshape(image_patches_val ,(len(X_val)*num_patches,h_patch,h_patch,3))
  total_label_patches_val = tf.reshape(label_patches_val ,(len(Y_val)*num_patches,h_patch,h_patch,1))

  print(total_image_patches_train.shape , total_label_patches_train.shape)
  print(total_image_patches_val.shape , total_label_patches_val.shape)

  return total_image_patches_train , total_label_patches_train , total_image_patches_val,total_label_patches_val

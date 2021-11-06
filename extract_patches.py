def extract_patches(images):
  # images refer to images and labels
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
      

  image_patches = tf.image.extract_patches(images, ksizes, strides, rates, padding)
  print(image_patches.shape)


  num_patches = image_patches.shape[1]*image_patches.shape[2]
  h_patch = ksize_rows

  I = tf.reshape(image_patches , (len(X)*num_patches,h_patch,h_patch,3))
  total_image_patches = tf.reshape(I ,(len(images)*num_patches,h_patch,h_patch,3))


  return total_image_patches

# so we shall do the same for ground-truth

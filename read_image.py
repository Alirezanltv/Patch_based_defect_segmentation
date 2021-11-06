import numpy as np
import cv2
import glob


def read_images(path_directory):

  image_file_path = path_directory
  image_file = glob.glob(image_file_path +'*.PNG')
  image_file.sort()
  X= []
  for img in image_file:
      image = cv2.imread(img)
      image = cv2.resize(image ,(448,448))
      # image equalization with clahe algorithm
      lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
      lab_planes = cv2.split(lab)
      clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(5,5))
      lab_planes[0] = clahe.apply(lab_planes[0])
      lab = cv2.merge(lab_planes)
      image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
      # ////end of clahe process
      image = image.astype('float32')
      image = image /255.
      # image = rgb2gray(image)
      image = image.reshape(448,448,3)
      X.append(image)
      
  X= np.array(X)
  return X

def read_labels(path_labels):

  label_file_path =  '/drive/My Drive/MY_DAGM2007/Class7/Train_labels/Label/'
  label_file = glob.glob(label_file_path +'*.PNG')
  label_file.sort()

  Y= []

  for img in label_file:
      label = cv2.imread(img)
      label = cv2.resize(label ,(448,448))
      label = label.astype('uint8')
      label = label / 255.
      label = label[:,:,0]
      label = label.reshape(448,448,1)
      # one_hot
  #     label = utils.to_categorical(label ,num_classes = 2)
      # //////
      Y.append(label)
  Y = np.array(Y)
  return Y

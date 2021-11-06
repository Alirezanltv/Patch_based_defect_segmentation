def train():
  batch_size = 49//4
  # batch_size =41
  train_num_classes = len(total_image_patches_train)
  steps = train_num_classes//batch_size+1

  val_num_classes = len(total_image_patches_val)

  model.compile(optimizer = RMSprop(lr = 1e-4) , loss = sm.losses.JaccardLoss(smooth=1) , metrics = [tf.keras.metrics.MeanIoU(num_classes=2)])

  from tensorflow.keras import callbacks

  model_checkpoint = callbacks.ModelCheckpoint('best_model.h5'
                                                        ,monitor ='val_loss',save_best_only =True)

  logger = callbacks.CSVLogger('best_model.log' , separator = ' ' )


  Callbacks = [model_checkpoint , logger]


  model.fit(total_image_patches_train, total_label_patches_train , epochs = 30, steps_per_epoch=steps,
                      validation_data=(total_image_patches_val , total_label_patches_val) ,validation_steps = val_num_classes//batch_size+1,
                      shuffle =True,verbose =1 , callbacks =Callbacks)
  return "GOOD LUCK"
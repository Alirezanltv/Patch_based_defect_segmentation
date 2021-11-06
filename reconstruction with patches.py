def reconstruction_with_patches(patches):
  ch = 1
  p = h_patch

  h = 448

  R = []
  pad =np.array([[0,0],[0,0]])

  for t in range(0,len(patches)):

      patches = tf.reshape(patches[t], [num_patches, p, p, ch])

      # Do processing on patches
      # Using patches here to reconstruct
      patches_proc = tf.reshape(patches,[1,h//p,h//p,p*p,ch])
      patches_proc = tf.split(patches_proc,p*p,3)

      patches_proc = tf.stack(patches_proc,axis=0)
      patches_proc = tf.reshape(patches_proc,[p*p,h//p,h//p,ch])

      reconstructed = tf.compat.v1.batch_to_space_nd(patches_proc,[p, p],pad)
      rec = tf.reshape(reconstructed , (h,h))
      
      R.append(rec)
  R = np.array(R)
  return R

 

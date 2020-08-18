import tensorflow as tf 


def multichol2inv(mat, n_mat, dtype=tf.float32):
    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.tile(tf.expand_dims(tf.eye(tf.shape(mat)[1], dtype=dtype), 
                                                      axis=0), 
                                       multiples=(n_mat,1,1) ) )
    invmat = tf.matmul(invlower, invlower, transpose_a=True)
    return invmat    


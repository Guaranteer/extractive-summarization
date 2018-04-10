import tensorflow as tf


item_num = 100
comp_num = 49
comp_dim = 2048
max_related_num = 20
batch_size = 100
user_dim = 512
D = 40

X = tf.placeholder(tf.float32, [item_num, comp_num, comp_dim],name='X')
Ri = tf.placeholder(tf.int32, [batch_size, max_related_num],name='Ri')
u = tf.placeholder(tf.float32, [batch_size,user_dim],name='u')



W2u = tf.Variable(tf.truncated_normal(shape=[user_dim,D], stddev=5e-2), dtype=tf.float32, name='W2u')
W2x = tf.Variable(tf.truncated_normal(shape=[comp_dim,D], stddev=5e-2), dtype=tf.float32, name='W2x')
W2 = tf.Variable(tf.truncated_normal(shape=[D,1], stddev=5e-2), dtype=tf.float32, name='W2')


xi = tf.nn.embedding_lookup(X,Ri)
print('xi',xi.shape)
xi = tf.reshape(xi,shape=[-1,comp_dim])
xb = tf.matmul(xi,W2x)
ub = tf.matmul(u,W2u)
print('ub',ub.shape)
ub = tf.tile(tf.expand_dims(tf.expand_dims(ub, [1]),[1]), tf.stack([1, max_related_num, comp_num, 1]))
ub = tf.reshape(ub,[-1,D])
a = tf.matmul(xb + ub, W2)
print('a',a.shape)
a = tf.reshape(a, [batch_size,max_related_num,comp_num])
print('a',a.shape)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
import tensorflow as tf
import numpy as np

# values of training data
training_set_inputs =np.array([[0,1,2],[0,0,2],[1,1,1],[1,0,1]])
training_set_outputs =np.array([[1],[0],[1],[0]])

# containers and operations
x = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.zeros([3, 1]))
B = tf.Variable(tf.zeros([1]))

yHat = tf.nn.sigmoid(tf.matmul(x, W) + B)
yLb = tf.placeholder(tf.float32, [None, 1])

learning_rate = 0.5
mean_square_loss = tf.reduce_mean(tf.square(yLb - yHat)) 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_square_loss)

# use session to execute graphs
sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
# or use this instead (for reading variables and loading initial value into the session):-
# init.run(session=sess)

# start training
for i in range(10000):
    sess.run(train_step, feed_dict={x: training_set_inputs, yLb: training_set_outputs})

# do prediction
x0=np.float32(np.array([[0.,1.,0.]]))   
y0=tf.nn.sigmoid(tf.matmul(x0,W) + B)

print('%.15f' % sess.run(y0))

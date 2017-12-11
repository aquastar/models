from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
import time
import tensorflow as tf
from scipy.stats import stats
from numpy import linalg as LA
from tutorials.image.mnist.gcn.models import GCN, MLP
from tutorials.image.mnist.gcn.utils import *


def add_perturbation(x, dim):
    x += np.float32(np.random.random([dim, 1]))
    return x


def get_fun_val(A, b, c, num=1000):
    for _ in xrange(num):
        x = np.random.random()
        func_val = 0.5 * np.dot(np.dot(np.transpose(x), A), x) + np.dot(np.dot(b), x) + c
    return


def getHessian(dim, val_A, val_b, val_c):
    # Each time getHessian is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs: A, b, and c.
        A = tf.placeholder(tf.float32, shape=[dim, dim])
        b = tf.placeholder(tf.float32, shape=[dim, 1])
        c = tf.placeholder(tf.float32, shape=[1])

        # Define our variable
        input_data = np.float32(np.random.random([dim, 1,100]))
        x = tf.Variable(input_data)
        xp = tf.Variable(tf.squeeze(x))
        x_perturb = tf.Variable(add_perturbation(input_data, dim))

        # x = tf.Variable(np.float32(np.repeat(1, dim).reshape(dim, 1)))

        # Construct the computational graph for quadratic function: f(x) = 1/2 * x^t A x + b^t x + c
        # set different parameters for A/b/c
        fx = 0.5 * tf.matmul(tf.matmul(tf.transpose(x), A), x) + tf.matmul(tf.transpose(b), x) + c
        fxp = 0.5 * tf.matmul(tf.matmul(tf.transpose(tf.reshape(xp, [dim, 1])), A), tf.reshape(xp, [dim, 1])) + \
              tf.matmul(tf.transpose(b), tf.reshape(xp, [dim, 1])) + \
              c
        fx_perturb = 0.5 * tf.matmul(tf.matmul(tf.transpose(x_perturb), A), x_perturb) + tf.matmul(tf.transpose(b),
                                                                                                   x_perturb) + c
        lst_solution = tf.matrix_inverse(tf.transpose(fx_perturb) * fx_perturb) * tf.transpose(fx_perturb) * fx

        # Get gradients of fx with repect to x
        dfx = tf.gradients(fx, x)[0]
        # Compute hessian
        for i in range(dim):
            # Take the i th value of the gradient vector dfx
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dfx, begin=[i, 0], size=[1, 1])
            # Feed it to tf.gradients to compute the second derivative.
            # Since x is a vector and dfx_i is a scalar, this will return a vector : [d(dfx_i) / dx_i , ... , d(dfx_n) / dx_n]
            ddfx_i = tf.gradients(dfx_i, x)[
                0]  # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            if i == 0:
                hess = ddfx_i
            else:
                hess = tf.concat([hess, ddfx_i], 1)
                ## Instead of doing this, you can just append each element to a list, and then do tf.pack(list_object) to get the hessian matrix too.
                ## I'll use this alternative in the second example.

        # directly calculate hession
        # hardHess = tf.hessians(fxp, xp)

        # Before we execute the graph, we need to initialize all the variables we defined
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:

            if 'session' in locals() and sess is not None:
                print('Close interactive session')
                sess.close()

            sess.run(init_op)
            # We need to feed actual values into the computational graph that we created above.
            feed_dict = {A: val_A, b: val_b, c: val_c}
            # sess.run() executes the graph. Here, "hess" will be calculated with the values in "feed_dict".
            x, hess, fx_r, fx = sess.run([x, hess, fx_perturb, fx], feed_dict)
            print('print calculated and real Hessian', hess, val_A)
            return x, fx, fx_r


def getHessianMLP(n_input, n_hidden, n_output):
    batch_size = 1
    # Each time getHessianMLP is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs and targets: x_input, y_target
        x_input = tf.placeholder(tf.float32, shape=[batch_size, n_input])
        y_target = tf.placeholder(tf.float32, shape=[batch_size, n_output])

        # Start constructing a computational graph for multilayer perceptron
        ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
        ### reshape it later according to each layer specification.
        parameters = tf.Variable(tf.squeeze
                                 (tf.concat([tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]),
                                             tf.truncated_normal([n_hidden * n_output, 1]), tf.zeros([n_output, 1])],
                                            0)))

        with tf.name_scope("hidden") as scope:
            idx_from = 0
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_input * n_hidden]),
                                 [n_input, n_hidden])
            idx_from = idx_from + n_input * n_hidden
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_hidden]),
                                [n_hidden])  # tf.Variable(tf.truncated_normal([n_hidden]))
            hidden = tf.matmul(x_input, weights) + biases
        with tf.name_scope("linear") as scope:
            idx_from = idx_from + n_hidden
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_hidden * n_output]),
                                 [n_hidden, n_output])
            idx_from = idx_from + n_hidden * n_output
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_output]), [n_output])
            output = tf.nn.softmax(tf.matmul(hidden, weights) + biases)
        # Define cross entropy loss
        loss = -tf.reduce_sum(y_target * tf.log(output))

        ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES
        ### because we are using g as our default graph inside the "with" scope.
        # Get trainable variables
        tvars = tf.trainable_variables()
        # Get gradients of loss with repect to parameters
        dloss_dw = tf.gradients(loss, tvars)[0]
        dim = dloss_dw.get_shape()

        trueHess = tf.hessians(loss, tvars)

        hess = []
        for i in range(dim[0].value):
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dloss_dw, begin=[i], size=[1])
            ddfx_i = tf.gradients(dfx_i, parameters)[
                0]  # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            hess.append(ddfx_i)
        hess = tf.squeeze(hess)

        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            input_x = np.random.random([batch_size, n_input])
            input_y = np.random.random([batch_size, n_output])

            feed_dict = {x_input: input_x, y_target: input_y}

            # print(sess.run(loss, feed_dict))
            # print(hess.get_shape())
            # start = time.time()
            # tf_hess = sess.run([trueHess], feed_dict)cd
            # end = time.time()
            # print end - start

            start = time.time()
            hess = sess.run([hess], feed_dict)
            print
            np.linalg.det(hess[0])
            end = time.time()
            print
            end - start


def train_comp(xx, yy, method='SGD'):
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.placeholder(tf.float32, shape=[None])

    # model
    W = tf.Variable(tf.truncated_normal([dim, dim], stddev=1))
    b = tf.Variable(tf.truncated_normal([dim, 1], stddev=1))
    c = tf.Variable(tf.truncated_normal([1], stddev=1))

    _y = tf.matmul(tf.matmul(tf.transpose(x), W), x) + tf.matmul(tf.transpose(b), x) + c

    # training and cost function
    cost_function = tf.reduce_mean(tf.square(y - _y))
    train_function = tf.train.GradientDescentOptimizer(1e-2).minimize(cost_function)

    # create a session
    sess = tf.Session()

    # train
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        sess.run(train_function, feed_dict={x: xx, y: yy})
        if i % 1000 == 0:
            print(sess.run(cost_function, feed_dict={x: xx, y: yy}))


def train_gcn(feat, adj, label):
    from itertools import cycle
    import matplotlib.pyplot as plt
    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'det', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'simu'
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(feat, adj, label)

    # Some preprocessing
    # features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [sparse_to_tuple(adj)]
        num_supports = 1
        model_func = GCN
        print('gcn')
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
        print('gcn_cheby')
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
        print('dense')
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    logdir = '/Users/danny/PycharmProjects/gcn/tflog'
    tb_write = tf.summary.FileWriter(logdir)
    tb_write.add_graph(sess.graph)

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders,
                                                  name=FLAGS.model)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # save the model

    return


if __name__ == '__main__':
    # Simulate 1: Ranking of function's Det(A) compared with estimated Det(A)
    dim = 3
    # when setting A matrix to symmetric, it would be the hessian matrix

    # Step 1: init A, and generate training data
    esti_arr = []
    real_arr = []
    val_A = 10 * np.float32(np.random.random([dim, dim]))
    val_A = (val_A + val_A.T) / 2
    val_b = np.float32(np.random.random([dim, 1]))
    val_c = [np.random.random()]

    x, fx, fx_r = getHessian(dim, val_A, val_b, val_c)

    # Step 2: try to get A/b/c by
    # SGD;
    # SGD + |det(hessian)|;
    # |det(hessian)|;
    # Hessian and other baselinse;
    train_comp(x, fx, 'SGD')

    # print(stats.pearsonr(esti_arr, real_arr), stats.spearmanr(esti_arr, real_arr))

    # train_gcn(esti_arr, esti_arr, real_arr)

    # 2.1 fix a A/b/c and test the G
    # val_A = 10 * np.float32(np.random.random([dim, dim]))
    # val_A = (val_A + val_A.T) / 2
    # val_b = np.float32(np.random.random([dim, 1]))
    # val_c = [1]
    # 2.2 generate training data
    # training = get_fun_val(val_A, val_b, val_c)


    # normal gradient, and novel plug G

    # getHessianMLP(n_input=2, n_hidden=2, n_output=2)

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # gvs = optimizer.compute_gradients(cost)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_op = optimizer.apply_gradients(capped_gvs)

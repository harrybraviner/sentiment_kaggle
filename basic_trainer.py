#! /usr/bin/python3

import utils, re
import numpy as np
import tensorflow as tf

def prepare_input(dictionary, phrase):
    words = re.split(r'\s|-', phrase.lower())
    word_vectors = [dictionary.try_get_word(w) for w in words if not(dictionary.try_get_word(w) is None)]
    if len(word_vectors) > 0:
        return np.average(word_vectors, axis=0)
    else:
        return None

def train_single_case(dataset, word_dictionary, optimizer, x, y, sess):
    phrase, sentiment = dataset.get_training_case()

    x_in = prepare_input(word_dictionary, phrase)
    # Let us know if we couldn't make sense of any of the words
    if x_in is None:
        #print("Unable to train on phrase: {}".format(phrase))
        return

    #y = np.zeros(5, dtype = np.float32)
    #y[sentiment] = 1.0

    optimizer.run(feed_dict = {x : np.array([x_in]), y : np.array([sentiment])}, session = sess)

def train_batch(dataset, word_dictionary, optimizer, x, y, sess, cross_entropy, N):
    data = [dataset.get_training_case() for i in range(N)]

    data_vec = [None] * len(data)
    for (i, (phrase, s)) in enumerate(data):
        vec = prepare_input(word_dictionary, phrase)
        if vec is None:
            #print("Unable to train on phrase: {}".format(phrase))
            None
        else:
            data_vec[i] = vec, s

    data_vec = [x for x in data_vec if not(x is None)]

    x_in = np.array([v for (v, _) in data_vec])
    y_in = np.array([s for (_, s) in data_vec])

    optimizer.run(feed_dict = {x : np.array(x_in), y : np.array(y_in)}, session = sess)
    ce_val = np.sum(cross_entropy.eval(feed_dict = {x : np.array(x_in), y : np.array(y_in)}, session = sess))

    return ce_val

def evaluate_validation(dataset, word_dictionary, x, y, sess, correct_count):
    batch_size = 64
    
    validation_set = dataset.get_validation_set()

    c_count = 0
    total_count = 0
    while validation_set != []:
        this_batch = validation_set[:batch_size]
        validation_set = validation_set[batch_size:]

        data_vec = [None] * len(this_batch)
        for (i, (phrase, s)) in enumerate(this_batch):
            vec = prepare_input(word_dictionary, phrase)
            if vec is None:
                #print("Unable to train on phrase: {}".format(phrase))
                None
            else:
                data_vec[i] = vec, s

        # For any sentiments where we couldn't use any of the words, we assume it's neutral
        sentiments_not_infered = [s for ((_, s), x) in zip(this_batch, data_vec) if x is None]
        #print("foo: {}".format(sentiments_not_infered))
        c_count += len([x for x in sentiments_not_infered if x == 2])
        total_count += len(sentiments_not_infered)

        data_vec = [x for x in data_vec if not(x is None)]

        x_in = np.array([v for (v, _) in data_vec])
        y_in = np.array([s for (_, s) in data_vec])

        c_count += correct_count.eval(feed_dict={x : np.array(x_in), y : np.array(y_in)}, session = sess)
        total_count += len(data_vec)

    frac_correct = c_count / total_count
            
    return frac_correct

def make_xavier_weights(shape):
    n_in, n_out = shape[0], shape[1]
    c = np.sqrt(6.0 / (n_in + n_out))
    return tf.Variable(tf.random_uniform(shape = shape, minval = -c, maxval = +c, dtype=tf.float32))

def make_bias(size):
    return tf.Variable(tf.constant(value = 0.0, shape = [size], dtype=tf.float32))

def main():

    dataset = utils.DataSet()
    word_dictionary = utils.GloveDictionary()

    dw = word_dictionary.dw
    #dw = 50
    n_h1 = 1024
    n_h2 = 1024
    n_classes = 5
    l2_cost = 1e-4

    x = tf.placeholder(dtype=tf.float32, shape = [None, dw])

    W1 = make_xavier_weights([dw, n_h1])
    b1 = make_bias(n_h1)
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = make_xavier_weights([n_h1, n_h2])
    b2 = make_bias(n_h2)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = make_xavier_weights([n_h2, n_classes])
    b3 = make_bias(n_classes)
    y_hat_logits = tf.matmul(h2, W3) + b3

    y = tf.placeholder(dtype=tf.int64, shape = [None])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = y_hat_logits)
    regularization = l2_cost * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    loss_quantity = cross_entropy + regularization

    y_hat_class = tf.argmax(y_hat_logits, axis=1)
    correct_count = tf.reduce_sum(tf.cast(tf.equal(y_hat_class, y), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer().minimize(loss_quantity)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    outfile = open('training.dat', 'wt')
    outfile.write("#iteration\tcross_entropy\tvalidation error\n")
    outfile.close()

    #train_single_case(dataset, word_dictionary, optimizer, x, y, sess)
    for i in range(40000):
        ce_val = train_batch(dataset, word_dictionary, optimizer, x, y, sess, cross_entropy, 64)

        if i%100 == 0:
            print("Iteration {}\nCross-entropy: {}\n".format(i, ce_val))
            frac_correct = evaluate_validation(dataset, word_dictionary, x, y, sess, correct_count)
            print("Frac correct on validation: {}\n".format(frac_correct))

            outfile = open('training.dat', 'at')
            outfile.write("{}\t{}\t{}\n".format(i, ce_val, frac_correct))
            outfile.close()

            

if __name__ == '__main__':
    main()

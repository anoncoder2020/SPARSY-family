from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from utils_cycle import *
from models import GCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('load_metis_data_train', 'cora_6cluster.npy', 'load clusters.')
flags.DEFINE_integer('num_cluster', 6, 'number of clusters.')
flags.DEFINE_integer('batch_size', 5, 'batch number of clusters.')
flags.DEFINE_float('percent_edges', 0.01, 'Initial learning rate.')
flags.DEFINE_float('percent_trained_node', 0.052,'for top_k_degree')

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 600, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0003, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 600, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layers', 2, 'num_layers.')



print("-----------------------------------------------")
print('batch_size', FLAGS.batch_size)
print("cluster: ", FLAGS.num_cluster)
print("percent_edges: ", FLAGS.percent_edges)
print("percent_trained_node: ", FLAGS.percent_trained_node)
print("num_layers: ", FLAGS.num_layers)
print("epochs", FLAGS.epochs)
print("learning_rate: ", FLAGS.learning_rate)
print("weight_decay", FLAGS.weight_decay)
print("-------------------------------------------------")


# Load data
features_init, graph, y, test_idx_range, labels, whole_adj = load_data(FLAGS.dataset)
cluster_for_train = np.load(FLAGS.load_metis_data_train)

features = preprocess_features(features_init)

if FLAGS.model == 'gcn':
    num_supports = 1
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# Create model
model = model_func(placeholders, input_dim=features[2][1], num_layers=FLAGS.num_layers, logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(t_dict, placeholders, method):
    t_test = time.time()
    total_loss = 0
    total_acc = 0
    total_num = 0
    if method == "val":
        for key in t_dict:
            e_feature = t_dict[key][1]
            e_support = t_dict[key][0]
            e_labels = t_dict[key][3]
            e_mask = t_dict[key][6]
            num_nodes = len(t_dict[key][9])
            num_data_b = np.sum(e_mask)
            feed_dict_val = construct_feed_dict(e_feature, e_support, e_labels, e_mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            total_loss += outs_val[0]* num_data_b
            total_acc += outs_val[1]* num_data_b
            total_num += num_nodes
        loss = total_loss / total_num
        acc = total_acc / total_num
        return loss, acc, (time.time() - t_test)
    if method == "test":
        for key in t_dict:
            e_feature = t_dict[key][1]
            e_support = t_dict[key][0]
            e_labels = t_dict[key][4]
            e_mask = t_dict[key][7]
            num_nodes = len(t_dict[key][10])
            num_data_b = np.sum(e_mask)
            feed_dict_val = construct_feed_dict(e_feature, e_support, e_labels, e_mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            total_loss += outs_val[0]* num_data_b
            total_acc += outs_val[1]* num_data_b
            total_num += num_nodes
        loss = total_loss/total_num
        acc = total_acc/total_num
        return loss, acc, (time.time() - t_test)



# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
dict_count = {}
store_dict = []
for epoch in range(FLAGS.epochs):
    whole_dict, chosen_id = minbatch_cycle(labels, y, test_idx_range, FLAGS.batch_size, FLAGS.num_cluster, epoch, cluster_for_train,
                                      whole_adj, features_init, FLAGS.percent_edges, FLAGS.percent_trained_node)
    for i in range(len(chosen_id)):
        if tuple(sorted(chosen_id[i])) in dict_count.keys():
            dict_count[tuple(sorted(chosen_id[i]))] += 1
        else:
            dict_count[tuple(sorted(chosen_id[i]))] = 0
            store_dict.append(whole_dict)

    # Training step
    t = time.time()
    for key in whole_dict:
        t1 = time.time()
        support = whole_dict[key][0]
        features = whole_dict[key][1]
        y_train = whole_dict[key][2]
        train_mask = whole_dict[key][5]

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})


        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)


    # Validation
    cost, acc, duration = evaluate(whole_dict, placeholders, method="val")
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break
print("distribution of shuffling of groups of clusters: ", dict_count)
print("Optimization Finished!")

# Testing
test_acc_list = []
for sub_dict in store_dict:
    test_cost, test_acc, test_duration = evaluate(sub_dict, placeholders, method="test")
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    test_acc_list.append(test_acc)
print("average: ", np.mean(test_acc_list))






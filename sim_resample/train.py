#encoding=utf-8
import pdb
import tensorflow as tf
import numpy as np
import os
import time
import sys
import datetime
import insurance_qa_data_helpers
from insqa_cnn import InsQACNN
import operator

#print tf.__version__

# Parameters
# ==================================================

# Data loading params

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "./data/cnews.train.txt.seg", "train data for Chinese.")#'/home/jwy/insuranceQA-cnn-lstm-master/insuranceQA/train'
tf.flags.DEFINE_string("dev_file", "./data/cnews.test.txt.seg", "val data for Chinese.")
tf.flags.DEFINE_string("test_file", "./data/cnews.test.txt.seg", "test data for Chinese.")
tf.flags.DEFINE_string("embedding_file", "/home/jwy/insuranceQA-cnn-lstm-master/insuranceQA/vectors.nobin", "embedding vectors file.")
tf.flags.DEFINE_integer("max_seq_len", 600, "Max document length(default: 600)")
tf.flags.DEFINE_string("model_version", "", "model version")
tf.flags.DEFINE_string("embedding_type", "word_embedding", "embedding type, word_embedding, char_embedding, subword_embedding")
tf.flags.DEFINE_integer("max_wordlen", 6, "subword max length(default: 6)")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("sample_type", "semi", "[semi] for semi-hard negative sample, [rand] for random sample (default: 'semi')")
tf.flags.DEFINE_float("min_margin", 0.01, "semi-hard sample minimum margin (default: 0.05)")
tf.flags.DEFINE_float("max_margin", 0.1, "semi-hard sample maximum margin (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("macro_batch_size", 128, "macro batch size for semi-hard sample (default: 256)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2500, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("is_test", False, "wether test model mode")
tf.flags.DEFINE_boolean("is_save", False, "wether save model mode")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")

raw, label_map = insurance_qa_data_helpers.read_raw(FLAGS.train_file)#get true query vs answer
max_seq_len = FLAGS.max_seq_len
if FLAGS.embedding_type == "char_embedding":
    vocab = insurance_qa_data_helpers.build_vocab_char(FLAGS.train_file)
elif FLAGS.embedding_type == "subword_embedding":
    vocab = insurance_qa_data_helpers.build_vocab_subword(FLAGS.train_file)
else:
    vocab = insurance_qa_data_helpers.build_vocab(FLAGS.train_file)#{word:id}

alist,avec_list = insurance_qa_data_helpers.read_alist(FLAGS.train_file, vocab, max_seq_len, FLAGS.embedding_type)#raw_label_list,id_label_list

testList = insurance_qa_data_helpers.load_test(FLAGS.dev_file, alist)
#for word,idx in vocab.items():
#   print word,idx
print "vocab size:",len(vocab)
#pdb.set_trace()
vectors = ''
print('seq_len', max_seq_len)
print("Load one...")

# Training
# ==================================================

with tf.Graph().as_default():
  with tf.device("/gpu:1"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = InsQACNN(
            sequence_length=max_seq_len,
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            embedding_type=FLAGS.embedding_type)#网络结构
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        #optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_version))
        

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print ("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Writing to {}\n".format(out_dir))
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

        def train_step(x_batch_1, x_batch_2, x_batch_3, indices_batch_1, indices_batch_2, indices_batch_3):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_1: x_batch_1,
              cnn.input_x_2: x_batch_2,
              cnn.input_x_3: x_batch_3,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            if FLAGS.embedding_type == "subword_embedding":
                feed_dict[cnn.indices_x_1] = indices_batch_1 
                feed_dict[cnn.indices_x_2] = indices_batch_2
                feed_dict[cnn.indices_x_3] = indices_batch_3
                feed_dict[cnn.subword_shape_1] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen])
                feed_dict[cnn.subword_shape_2] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen]) 
                feed_dict[cnn.subword_shape_3] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen])
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(testList):
          scoreList = []
          i = int(0)
          while True:
              x_test_1, x_test_2, indices_test_1, indices_test_2, tag_list = insurance_qa_data_helpers.load_data_val(testList, vocab, i, FLAGS.batch_size, max_seq_len, FLAGS.embedding_type)
              feed_dict = {
                cnn.input_x_1: x_test_1,
                cnn.input_x_2: x_test_2,
                cnn.input_x_3: x_test_2,
                cnn.dropout_keep_prob: 1.0
              }
              if FLAGS.embedding_type == "subword_embedding":
                    feed_dict[cnn.indices_x_1] = indices_test_1
                    feed_dict[cnn.indices_x_2] = indices_test_2
                    feed_dict[cnn.indices_x_3] = indices_test_2
                    feed_dict[cnn.subword_shape_1] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen])
                    feed_dict[cnn.subword_shape_2] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen])
                    feed_dict[cnn.subword_shape_3] = np.array([FLAGS.batch_size*FLAGS.max_seq_len,FLAGS.max_wordlen])
              batch_scores = sess.run([cnn.cos_12], feed_dict)
              for score in batch_scores[0]:
                  scoreList.append(score)
              i += FLAGS.batch_size
              if i >= len(testList):
                  break
          assert len(scoreList) >= len(testList)
          acc,pred_list = max_eval(testList, scoreList)
          print "total_cnt:",len(pred_list)
          print "acc:",acc

          return acc,pred_list
        
        def max_eval(testraw_list, score_list):
            '''
            desc:计算准确率

            input:
                testraw_list:测试集
                score_list:预测的结果值
            return:
                acc:准确率
                pred_list:模型预测的结果值
            '''
            max_cand = {}
            ans_map = {}
            for inx in xrange(len(testraw_list)):
                tag, label, query = testraw_list[inx].split('\t')
                if tag == "1":
                    ans_map[query] = label#真实正样例的结果
                if query not in max_cand or score_list[inx] > max_cand[query][0]:
                    max_cand[query] = (score_list[inx],label)#模型预测的结果
            acc_cnt = 0
            pred_list = []
            for query in max_cand:
                is_right = False
                if query in ans_map and max_cand[query][1] == ans_map[query]:
                    is_right = True
                    acc_cnt += 1
                pred_list.append((query, max_cand[query][1], str(max_cand[query][0]), ans_map.get(query,"")))
            acc = 1.0 * acc_cnt / len(max_cand)
            return acc, pred_list
        
        if FLAGS.is_test:#测试时
            testList = insurance_qa_data_helpers.load_test(FLAGS.test_file, alist)
            dev_step(testList)
            sys.exit(0)
        # Generate batches
        # Training loop. For each batch...
        best_acc = 0.0
        for i in range(FLAGS.num_epochs):
            try:
                if FLAGS.sample_type == 'rand':
                    '''
                    随机采样
                    '''
                    x_batch_1, x_batch_2, x_batch_3, indices_batch_1, indices_batch_2, indices_batch_3 = insurance_qa_data_helpers.rand_sample(vocab, alist, raw, FLAGS.batch_size, max_seq_len, FLAGS.embedding_type)
                else:
                    '''
                    semihard采样
                    '''
                    x_batch_1, x_batch_2, x_batch_3, indices_batch_1, indices_batch_2, indices_batch_3 = insurance_qa_data_helpers.semihard_sample(sess, cnn, vocab, raw, FLAGS.batch_size, max_seq_len, avec_list, label_map, alist, FLAGS.macro_batch_size, FLAGS.min_margin, FLAGS.max_margin, FLAGS.embedding_type, FLAGS.max_wordlen)
                #pdb.set_trace()
                train_step(x_batch_1, x_batch_2, x_batch_3, indices_batch_1, indices_batch_2, indices_batch_3)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:", current_step)
                    cur_acc, _ = dev_step(testList)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    if cur_acc >= best_acc:
                        best_acc = cur_acc
                    else:
                        print("current accuracy drop and stop train..\n")
                        sys.exit(0)
                    print("")
                #if current_step % FLAGS.checkpoint_every == 0:
                #   path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #   print("Saved model checkpoint to {}\n".format(path))
            except Exception as e:
                print(e)

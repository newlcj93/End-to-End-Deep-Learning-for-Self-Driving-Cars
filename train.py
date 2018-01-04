'''
You are going to train the CNN model here.

'''
import os
import tensorflow as tf
from model import Model
import numpy as np
from load_data import LoadTrainBatch,LoadValBatch
import datetime

tf.flags.DEFINE_float("learning_rate", 1e-4, "learning rate (default: 1e-4)")
tf.flags.DEFINE_integer("batch_size",64, "batch_size(default: 64)")
tf.flags.DEFINE_integer("max_iter_num",10000, "max_iter_num(default: 10000)")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep_prob(default: 0.5)")
tf.flags.DEFINE_float("lambda_l2", 0.1, "lambda_l2(default: 0.1)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

LOGDIR = './save'
CKPT_FILE = './save/model.ckpt'
TENSORBOARD_LOG = './logs'

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(TENSORBOARD_LOG):
    os.makedirs(TENSORBOARD_LOG)
checkpoint_prefix = os.path.join(LOGDIR, "model")

with tf.device('/gpu:0'):
    config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess.as_default():
        model = Model(FLAGS.lambda_l2)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss,global_step=global_step)
        train_summary = tf.summary.scalar("train_loss", model.loss)
        val_summary = tf.summary.scalar("val_loss", model.loss)
        summary_writer = tf.summary.FileWriter(TENSORBOARD_LOG, sess.graph)
        saver = tf.train.Saver()
        if os.path.isfile(CKPT_FILE):
            saver.restore(sess, CKPT_FILE)
        else:
            sess.run(tf.global_variables_initializer())

    batch_size = FLAGS.batch_size
    max_iter_num = FLAGS.max_iter_num
    min_val_loss = 1e3
    for i in xrange(max_iter_num):
        x_train,y_train = LoadTrainBatch(batch_size)
        _, train_loss_summary, loss, step = sess.run([train_op,train_summary,model.loss, global_step],
                                    feed_dict = {model.x:x_train,model.y_:y_train,
                                                 model.keep_prob:FLAGS.keep_prob})
        time = datetime.datetime.now().isoformat()
        print("%s step: %d, train loss: %g" % (time,step, loss))
        if step % 100 == 0:
            summary_writer.add_summary(train_loss_summary, step)
            x_val,y_val = LoadValBatch(batch_size)
            val_loss_summary, loss,step = sess.run([val_summary, model.loss,global_step],
                                        feed_dict={model.x: x_val, model.y_: y_val,
                                                   model.keep_prob: 1.0})
            summary_writer.add_summary(val_loss_summary, step)
            time = datetime.datetime.now().isoformat()
            print("%s step: %d, val loss: %g" % (time, step, loss))
            if loss < min_val_loss:
                min_val_loss = loss
                saver.save(sess, checkpoint_prefix,global_step=step)
                print("Saved model {} with val_loss={}\n".format(step, min_val_loss))


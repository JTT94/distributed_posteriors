import argparse
import sys
from time import  sleep
import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        with tf.device("/job:ps/task:0"):
            var = tf.Variable([0.0,0.0], name='var')
        sess = tf.Session(target=server.target)
        sess.run(tf.global_variables_initializer())
        server.join()


    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            with tf.device("/job:ps/task:0"):
                var = tf.Variable([0.0, 0.0], name='var')

            sess = tf.Session(target=server.target)
            sess.run(tf.global_variables_initializer())
            print('Worker begin update')
            for _ in range(1000):
                print('Worker {0}'.format(FLAGS.task_index*2-1))
                if FLAGS.task_index == 0:
                    sess.run(var.assign_add([0, FLAGS.task_index*2-1 + FLAGS.task_index*3], use_locking=True))
                else:
                    sess.run(var.assign_add([FLAGS.task_index * 2 - 1 + FLAGS.task_index * 3,0], use_locking=True))
                print(sess.run(var))
        print('Worker blocking')
        server.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

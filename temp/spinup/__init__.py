# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Algorithms
from temp.spinup.algos.tf1.td3.td3 import td3 as td3_tf1

# Loggers

# Version

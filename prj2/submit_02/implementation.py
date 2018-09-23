import tensorflow as tf
# import string
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_CLASS = 2   # number of classes
NUM_HIDDEN_UNIT = 32   # number of hidden unit
L_RATE = 0.01  # learning rate
CELL_MODEL = "LSTM"
# CELL_MODEL = "GRU"
# CELL_MODEL = "BIO"


stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_string = review.lower()
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    table = processed_string.maketrans("", "", punctuation)
    processed_string = processed_string.translate(table)
    processed_string = processed_string.split()
    processed_review = []
    for item in processed_string:
        if item not in stop_words:
            processed_review.append(item)
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    # tf.reset_default_graph()
    # shape: BATCH_SIZE * MAX_WORDS_IN_REVIEW * EMBEDDING_SIZE
    input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
            name="input_data")
    # shape: BATCH_SIZE * NUM_CLASS
    labels = tf.placeholder(
            dtype=tf.float32,
            shape=[BATCH_SIZE, NUM_CLASS],
            name="labels")
    # shape: Unknown
    dropout_keep_prob = tf.placeholder_with_default(
            1.0,
            shape = (),
            name='dropout_keep_prob')

    if CELL_MODEL == "LSTM":
        # output size: NUM_HIDDEN_UNIT
        cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_UNIT)
        # cell = tf.contrib.rnn.GRUCell(NUM_HIDDEN_UNIT)
        cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                output_keep_prob=dropout_keep_prob)
        # outputs shape: BATCH_SIZE * MAX_WORDS_IN_REVIEW * NUM_HIDDEN_UNIT
        # state shape: BATCH_SIZE * NUM_HIDDEN_UNIT
        outputs, state = tf.nn.dynamic_rnn(
                cell = cell, 
                inputs = input_data,
                dtype=tf.float32)
        # shape: BATCH_SIZE * NUM_HIDDEN_UNIT
        last_output = state[1]
        # shape NUM_HIDDEN_UNIT * NUM_CLASS
        weight = tf.Variable(tf.truncated_normal(
                [NUM_HIDDEN_UNIT, NUM_CLASS]))
    elif CELL_MODEL == "GRU":
        cell = tf.contrib.rnn.GRUCell(NUM_HIDDEN_UNIT)
        cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                output_keep_prob=dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(
                cell = cell, 
                inputs = input_data,
                dtype=tf.float32)
        last_output = tf.unstack(
                tf.transpose(outputs, [1,0,2]))[-1]
        # shape NUM_HIDDEN_UNIT * NUM_CLASS
        weight = tf.Variable(tf.truncated_normal(
                [NUM_HIDDEN_UNIT, NUM_CLASS]))
    else:           # BIO
        cell_fw = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_UNIT)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_UNIT)
        cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_fw,
                output_keep_prob=dropout_keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_bw,
                output_keep_prob=dropout_keep_prob)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell_fw,
                cell_bw = cell_bw,
                inputs = input_data,
                dtype = tf.float32)
        outputs = tf.concat((output_fw, output_bw),2)
        last_output = outputs[:,-1,:]
        # shape NUM_HIDDEN_UNIT * NUM_CLASS
        weight = tf.Variable(tf.truncated_normal(
                    [NUM_HIDDEN_UNIT * 2, NUM_CLASS]))

    # shape ? * NUM_CLASS
    bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASS]))
    # shape: BATCH_SIZE* NUM_CLASS
    logits = (tf.matmul(last_output, weight) + bias)

    # shape: BATCH_SIZE * ?
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels),
    loss = tf.reduce_mean(cross_entropy, name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=L_RATE).minimize(loss)
    # shape: BATCH_SIZE * ?
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_pred = tf.cast(correct_pred, tf.float32)
    Accuracy = tf.reduce_mean(correct_pred, name="accuracy")
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
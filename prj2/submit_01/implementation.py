import tensorflow as tf
# import string
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_CLASS = 2   # number of classes
NUM_HIDDEN_UNIT = 128   # number of hidden unit

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
    # Input placeholder: name="input_data"
    # BATCH_SIZE * MAX_WORDS_IN_REVIEW * EMBEDDING_SIZE
    input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
            name="input_data")
    # labels placeholder: name="labels"
    # BATCH_SIZE * NUM_CLASS
    labels = tf.placeholder(
            dtype=tf.float32,
            shape=[BATCH_SIZE, NUM_CLASS],
            name="labels")
    # dropout placeholder
    dropout_keep_prob = tf.placeholder_with_default(
            1.0,
            shape = (),
            name='dropout_keep_prob')

    # define RNN
    # weights = {
    #     'in': tf.Variable(tf.random_normal(
    #                 [MAX_WORDS_IN_REVIEW, NUM_HIDDEN_UNIT])),
    #     'out': tf.Variable(tf.random_normal(
    #                 [NUM_HIDDEN_UNIT, NUM_CLASS]))
    # }
    # biases = {
    #     'in': tf.Variable(tf.constant(0.1, shape=[NUM_HIDDEN_UNIT, ])),
    #     'out': tf.Variable(tf.constant(0.1, shape=[NUM_CLASS, ]))
    # }        
    # input layer
    # input_data_2D = tf.reshape(input_data, [-1, EMBEDDING_SIZE])
    # input_data_2D = tf.matmul(
    #         input_data_2D,
    #         weights['in']) + biases['in']
    # input_data_3D = tf.reshape(input_data_2D,
    #         [-1, MAX_WORDS_IN_REVIEW, NUM_HIDDEN_UNIT])

    # hidden layer
    # cell = tf.nn.rnn_cell.BasicLSTMCell(
    #         NUM_HIDDEN_UNIT,
    #         forget_bias=1.0,
    #         state_is_tuple=True)
    # init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    # outputs, state = tf.nn.dynamic_rnn(
    #         cell,
    #         input_data_3D,
    #         initial_state=init_state,
    #         time_major=False)
    # # output layer
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # logits = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
    weight = tf.Variable(tf.truncated_normal(
                [NUM_HIDDEN_UNIT, NUM_CLASS]))
    bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASS]))

    cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_UNIT)
    cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            output_keep_prob=dropout_keep_prob)
    
    outputs, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    
    logits = (tf.matmul(outputs, weight) + bias)

    # loss tensor: name="loss"
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels),
                name='loss')
    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # accuracy tensor: name="accuracy"
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32),
            name="accuracy")


    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss


if __name__ == "__main__":
    """
    for test only
    """
    # test preprocess
    # with open('./data/train/pos/2_7.txt', "r") as f:
    #     review = f.read()
    # print("\n", review, "\n")
    # print(preprocess(review))
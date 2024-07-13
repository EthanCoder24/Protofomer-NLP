# Protoformer in Tensorflow

Here is the tensorflow implementaiton for this paper. I got best results over the IMDB dataset from this paper. Hope it helps.


## Setup
Make sure to have the necessary libraries installed:

Copy code
pip install tensorflow numpy pandas
Data Preparation
Download and prepare the IMDB dataset:
# Hyperparameter Tuning
max_features = 20000
maxlen = 200
Dropout = 0.4

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

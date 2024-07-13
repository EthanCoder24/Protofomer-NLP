# =============================================================================
# Model 
# =============================================================================
import tensorflow as tf
from transformers import TFDistilBertModel, TFBertModel, TFRobertaModel
from transformers import DistilBertTokenizer, BertTokenizer, RobertaTokenizer
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import numpy as np

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 1e-05

# Define BERT-based models

class DistillBERT(Model):
    def __init__(self, num_classes):
        super(DistillBERT, self).__init__()
        self.bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = layers.Dropout(0.6)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        dropout_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        classifier_output = self.classifier(dropout_output)
        return classifier_output

class BERT(Model):
    def __init__(self, num_classes):
        super(BERT, self).__init__()
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.dropout = layers.Dropout(0.6)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        dropout_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        classifier_output = self.classifier(dropout_output)
        return classifier_output

class RoBERTA(Model):
    def __init__(self, num_classes):
        super(RoBERTA, self).__init__()
        self.bert = TFRobertaModel.from_pretrained("roberta-base")
        self.dropout = layers.Dropout(0.6)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        dropout_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        classifier_output = self.classifier(dropout_output)
        return classifier_output

# Prepare Data
class BertDataFormat(tf.keras.utils.Sequence):
    def __init__(self, dataframe, tokenizer, max_len, batch_size):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.dataframe[idx * self.batch_size: (idx + 1) * self.batch_size]
        inputs = self.tokenizer(
            batch_data['doc'].tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="tf"
        )
        targets = tf.convert_to_tensor(batch_data['labels'].tolist())
        return (inputs['input_ids'], inputs['attention_mask']), targets

# Assuming df_profile, train_df, and test_df are already defined
num_classes = len(df_profile.labels.unique())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
baseline_model = BERT(num_classes)

train_data_gen = BertDataFormat(train_df, tokenizer, MAX_LEN, TRAIN_BATCH_SIZE)
test_data_gen = BertDataFormat(test_df, tokenizer, MAX_LEN, VALID_BATCH_SIZE)

# Compile model
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train model
history = baseline_model.fit(
    train_data_gen,
    validation_data=test_data_gen,
    epochs=EPOCHS
)

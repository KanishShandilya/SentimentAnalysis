import tensorflow as tf
from tensorflow.keras.layers import Embedding,Flatten,Dense,Dropout


class Model:

    def __init__(self,vocab_size,embedding_dim,max_length,num_classes):
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.max_length=max_length
        self.num_classes=num_classes

    def get_model(self):
        model=tf.keras.Sequential([
            Embedding(self.vocab_size,self.embedding_dim,input_length=self.max_length),
            Flatten(),
            Dense(16,activation='relu'),
            Dropout(0.2),
            Dense(16,activation='relu'),
            Dense(self.num_classes,activation='softmax')
        ])
        return model

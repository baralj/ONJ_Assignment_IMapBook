import csv
import os.path
from keras.layers import Embedding, LSTM, Dropout, Input, dot, subtract
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences
from keras.backend import abs
from keras.models import Model

sents1 = []
sents2 = []
scores = []

VOCAB_SIZE = 5000
EMB_DIM = 64
LSTM_DIM = 32
DROPOUT_RATIO = 0.5
BATCH_SIZE = 32

def cosine_loss(y_true, y_pred):
    return abs(subtract([y_true, y_pred]))

i = 0
for dirpath, dirnames, filenames in os.walk("Train_Data"):
    for filename in [f for f in filenames if f.endswith(".tsv")]:
        with open(os.path.join(dirpath, filename),encoding="utf8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    scores.append(float(row[0]) / 5.0)
                except ValueError:
                    continue
                sents1.append(hashing_trick(row[1], VOCAB_SIZE, hash_function='md5'))
                sents2.append(hashing_trick(row[2], VOCAB_SIZE, hash_function='md5'))

sents1_pad = pad_sequences(sents1, maxlen=50, padding="post")
sents2_pad = pad_sequences(sents2, maxlen=50, padding="post")

input1 = Input(shape=(50,))
input2 = Input(shape=(50,))

emb = Embedding(VOCAB_SIZE, EMB_DIM)
lstm_first = LSTM(LSTM_DIM, return_sequences=True)
drop = Dropout(DROPOUT_RATIO)
lstm_second = LSTM(LSTM_DIM)

emb1 = emb(input1)
lstm1_1 = lstm_first(emb1)
drop1 = drop(lstm1_1)
lstm1_2 = lstm_second(drop1)

emb2 = emb(input2)
lstm2_1 = lstm_first(emb2)
drop2 = drop(lstm2_1)
lstm2_2 = lstm_second(drop2)

cosine = dot([lstm1_2, lstm2_2], axes=-1, normalize=True)

model = Model(inputs=[input1, input2], outputs=[cosine])
print(model.summary())
model.compile(loss=cosine_loss,
              optimizer='adam',
              metrics=['accuracy'])

model.fit([sents1_pad, sents2_pad], scores, epochs= 50, batch_size=32)
model.save('model_lstm_v2_eph50.h5', overwrite=True)
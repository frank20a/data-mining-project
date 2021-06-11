import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


WORD_LIMIT = 100        # Delete x_y.pickle AND dataset.pickle when changed
WORD_DIMS = 200         # Place GloVe file in data and choose between 50, 100, 200, 300


class Health(pd.DataFrame):
    def __init__(self, filename='./data/spam_or_not_spam.csv'):
        # Initialize object
        super().__init__(pd.read_csv(filename))
        self.fillna('', inplace=True)

        # Initialize members
        self.vocab = None

        # Create Neural Network
        self.net = self.__getModel__()

        # Split to train/test
        try:
            with open("./data/x_y.pickle", 'rb') as file:
                self.x_train, self.x_test, self.y_train, self.y_test = pickle.load(file)
                print("Train/Test data loaded from pickle!")
        except FileNotFoundError:
            print("File not found... Splitting data to train/test!")

            # Dataset pre-processing
            self.__setEmails__()

            x = np.asarray(list(self.email)).astype('float32')
            y = np.array(self.label)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.33)
            with open("./data/x_y.pickle", 'wb') as file:
                pickle.dump((self.x_train, self.x_test, self.y_train, self.y_test), file)

    def __getVocab__(self):
        try:
            with open('data/vocabulary.pickle', 'rb') as file:
                vocab = pickle.load(file)
                print("Got vocab from pickle file!")
        except FileNotFoundError:
            print("Pickle file not found... Creating vocab from GloVe!")

            # Create list of useful words
            voc = []
            for email in self.email:
                for word in email.split():
                    if word not in voc: voc.append(word)

            # Save vectors of useful words
            with open(f'./data/glove.6B.{WORD_DIMS}d.txt', 'r', encoding='utf-8') as file:
                vocab = {}
                for line in file.readlines():
                    line = line.split()
                    if line[0] in voc:
                        vocab[line[0]] = list(map(float, line[1:]))
            with open('./data/vocabulary.pickle', 'wb') as file:
                pickle.dump(vocab, file)

        return vocab

    def __getModel__(self, num_lstm=int(WORD_LIMIT/2)):
        model = Sequential(name="SpamNet")

        model.add(LSTM(num_lstm, name='LSTM', input_shape=(WORD_LIMIT, WORD_DIMS)))
        model.add(Dense(32, name='Dense_1', activation='relu'))
        model.add(Dropout(0.2, name='Dropout_1'))
        model.add(Dense(16, name='Dense_2', activation='relu'))
        model.add(Dropout(0.05, name='Dropout_2'))
        model.add(Dense(1, name='Output', activation="sigmoid"))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=["binary_accuracy"])
        return model

    def __setEmails__(self):
        try:
            with open('./data/dataset.pickle', 'rb') as file:
                self.email = pickle.load(file)
                print("Emails loaded from pickle!")
        except FileNotFoundError:
            print("Pickle file not found... Preprocessing emails!")

            # Loading vocabulary
            self.vocab = self.__getVocab__()
            for i in range(len(self)):
                email = []
                for word in self.email[i].lower().split():
                    if word in self.vocab.keys(): email.append(self.vocab[word])
                if len(email) >= WORD_LIMIT:
                    email = email[:WORD_LIMIT]
                else:
                    email += [list(np.zeros(WORD_DIMS))] * (WORD_LIMIT - len(email))
                self.email[i] = email

            with open('./data/dataset.pickle', 'wb') as file:
                pickle.dump(self.email, file)

    def train(self):
        self.net.summary()
        print("Initiated training...")
        self.net.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), batch_size=64, epochs=50,
                     verbose=1)

        return self

    def get_scores(self):
        y_test = [[i] for i in self.y_test]
        y_pred = (self.net.predict(self.x_test) > .5).astype('int32')

        return f1_score(y_test, y_pred, average='macro'), \
               precision_score(y_test, y_pred, average='macro'), \
               recall_score(y_test, y_pred, average='macro')

    def predict(self, text):
        tr = {0: 'ham', 1: 'spam'}

        email = []
        if self.vocab is None: self.vocab = self.__getVocab__()
        for word in text.lower().split():
            if word in self.vocab.keys(): email.append(self.vocab[word])
        if len(email) >= WORD_LIMIT:
            email = email[:WORD_LIMIT]
        else:
            email += [list(np.zeros(WORD_DIMS))] * (WORD_LIMIT - len(email))

        return tr[(self.net.predict(np.asarray([email]).astype('float32')) > .5).astype('int32')[0][0]]


A = Health()
scores = A.train().get_scores()
print()
print("F1 score  ->  ".rjust(12) + "%2.2f%%" % (scores[0] * 100))
print("Precision  ->  ".rjust(12) + "%2.2f%%" % (scores[1] * 100))
print("Recall  ->  ".rjust(12) + "%2.2f%%" % (scores[2] * 100))

while True:
    print(A.predict(input()))

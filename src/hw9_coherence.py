import os
import sys
import json
import codecs
import nltk
import numpy as np
import sklearn
from collections import defaultdict
from sklearn.metrics import f1_score

# load in arguments
glove_embedding_file = sys.argv[1]
relation_training_data_file = sys.argv[2]
relation_testing_data_file = sys.argv[3]
training_vector_file = sys.argv[4]
testing_vector_file = sys.argv[5]
output_file = sys.argv[6]

# Read in Glove embedding vectors
glove_raw = open(os.path.join(os.path.dirname(__file__), glove_embedding_file), 'r').read().split("\n")[:-1]

# format Glove data
glove = defaultdict(lambda: np.zeros(50, ))
for entry in glove_raw:
    listified = entry.split()
    key = listified[0]
    val = np.array([float(x) for x in listified[1:]])
    glove[key] = val

# Load training and test coherence relation classification data
training_cr = codecs.open(relation_training_data_file, encoding='utf8')
training_relations = [json.loads(x) for x in training_cr]

testing_cr = codecs.open(relation_testing_data_file, encoding='utf8')
testing_relations = [json.loads(x) for x in testing_cr]


# function that takes a list of tokens and returns a vector of vectors, each row_i corresponding to the i-th token
def cum_vector(tokenized):
    listified = []
    for word in tokenized:
        listified.append(glove[word.lower()])
    vector = np.array(listified)
    return vector


# function that takes relation classification data and returns list of classification vectors per instance
def classifications(relation):
    span = []
    labels = []
    # For each shallow discourse parsing instance:
    # TODO: remove constriants
    for instance in relation:
        sense = instance['Sense'][0].split(".")[0]
        labels.append(sense)

        # For Arg1 and Arg2, tokenize the raw text, ideally using NTLK.word_tokenize()
        a1 = nltk.word_tokenize(instance['Arg1']['RawText'])
        a2 = nltk.word_tokenize(instance['Arg2']['RawText'])

        # Using the corresponding Glove embeddings of the tokens, create averaged vector representation of each Arg
        avg_a1 = np.mean(cum_vector(a1), axis=0)
        avg_a2 = np.mean(cum_vector(a2), axis=0)

        # Concatenate the Arg1 and Arg2 representation to make the classification vector
        classification_vec = np.concatenate([avg_a1, avg_a2])
        span.append(classification_vec)
    return np.array(span), np.array(labels)


# Creating training and test classification vectors from the data
train_classified = classifications(training_relations)
train_span = train_classified[0]
train_labels = train_classified[1]

test_classified = classifications(testing_relations)
test_span = test_classified[0]
test_labels = test_classified[1]

# Write the training and test instances to respective files in csv format
with open(training_vector_file, 'w') as train:
    for i in range(len(train_labels)):
        for num in train_span[i]:
            train.write(str(num) + ",")
        # with the sense of the instance as the last element in each line
        train.write(train_labels[i] + "\n")

with open(testing_vector_file, 'w') as test:
    for i in range(len(test_labels)):
        for num in test_span[i]:
            test.write(str(num) + ",")
        # with the sense of the instance as the last element in each line
        test.write(test_labels[i] + "\n")

# Train a classifier on the training instances
classifier = sklearn.svm.SVC(decision_function_shape='ovr')
classifier.fit(train_span, train_labels)

# Test on the test instances
predictions = classifier.predict(test_span)
class_f1s = f1_score(test_labels, predictions, average=None,
                     labels=["Expansion", "Comparison", "Contingency", "Temporal"])

# Writing to the output file
with open(output_file, 'w') as op:
    # The overall per-class F-measure
    op.write(str(class_f1s) + "\n")
    # For each test instance: true_label\tpredicted_label
    for i in range(len(predictions)):
        op.write(str(test_labels[i]) + "\t" + str(predictions[i]) + "\n")

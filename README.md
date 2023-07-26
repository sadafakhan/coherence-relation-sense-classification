# coherence-relation-sense-classification
```coherence-relation-sense-classification``` performs coherence relation sense classification by training a classifier on Glove embedding vectors.

Args: 
* ```glove_embedding_file```: path to file containing Glove embeddings 
* ```<relation_training_data_file>```: path to file containing json-formatted training data
* ```<relation_testing_data_file>```: path to file containing json-formatted testing data

Returns: 
* ```<training_vector_file>```: path to csv output file for training vector representations
* ```<testing_vector_file>```: path to csv output file for testing vector representations
* ```<output_filename>```: path to output file with classification results

To run: 
```
src/hw9_coherence.sh input/glove.6B.50d.txt input/relations_train.json input/relations_test.json output/hw9_training_vectors.txt output/hw9_testing_vectors.txt output/hw9_output.txt
```

HW9 OF LING571 (12/13/2021)
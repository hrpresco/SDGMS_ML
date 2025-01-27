import pandas as pd
import tensorflow as tf
import tempfile
import os

def convert_data_to_record_types():
    test_csv = "/users/haydenprescott/documents/test.csv"
    record_filepath = test_path = os.path.join(os.getcwd(), "test.tfrecords")
    test_dataset = pd.read_csv(test_csv).values
    with tf.compat.v1.python_io.TFRecordWriter(record_filepath) as writer:
        for row in test_dataset:
            features, label = row[:-1], row[-1]
            example_data_piece = tf.train.Example()
            example_data_piece.features.feature["features"].float_list.value.extend(features)
            example_data_piece.features.feature["label"].float_list.value.append(label)
            writer.write(example_data_piece.SerializeToString())
            #print (example_data_piece)
    return record_filepath


record_types = convert_data_to_record_types()
print(record_types)
#contents = contents.read()
#print(contents)
import pandas as pd
import tensorflow as tf
import tempfile
import os
import numpy as np
import mendeleev as ptable
from pathlib import Path


class molecular_graph:
    def __init__(self, orbital_overlaps, n_atoms, maximum_s_el, maximum_p_el, maximum_d_el, maximum_f_el):
        self.n_atoms = n_atoms
        self.orbital_overlaps = orbital_overlaps
        self.maximum_s_el = maximum_s_el
        self.maximum_p_el = maximum_p_el
        self.maximum_d_el = maximum_d_el
        self.maximum_f_el = maximum_f_el

    def build_adj_matrix(self, n_atoms, orbital_overlaps):
        self.labeled_atom_list = np.arange(0, n_atoms)
        self.adj_matrix = np.array([])
        self.adj_matrix = self.adj_matrix[np.newaxis, :]
        #classifying all atoms in complex as either engaging in orbital overlap (bonding M.O) or not (nonbonding state), starting from atom 1 to the nth atom in the system:
        i = 0
        j = 0
        for i in np.arange(0, n_atoms):
            ith_adjacency_row = np.zeros(n_atoms)
            ith_adjacecy_row = list(ith_adjacency_row)
            for j in np.arange(0, len(orbital_overlaps)):
                if str(i+1) in orbital_overlaps[j]:
                    if orbital_overlaps[j][0] == str(i+1):
                        overlapped_adjacent = int(orbital_overlaps[j][2])
                        ith_adjacency_row[overlapped_adjacent - 1] = 1
                j += 1
            self.adj_matrix = np.append(self.adj_matrix, ith_adjacency_row)
            i += 1
        self.adj_matrix = np.reshape(self.adj_matrix, (n_atoms, n_atoms))
        return self.adj_matrix
        
def convert_data_to_records():
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
            print(example_data_piece)
    return record_filepath


record_types = convert_data_to_records()
print(record_types)
#contents = contents.read()
#print(contents)

csv_filepath = "/users/haydenprescott/documents/test.csv"
record_filename = "test_data.tfrecord" 
record_filepath = "/users/haydenprescott/test_data.tfrecord"

def initialize_dataset(csv_filepath):
    records_dict = {}
    r_set = np.array([])
    ACSF_set = np.array([])
    z_set = np.array([])
    d1_set = np.array([])
    d2_set = np.array([])
    d3_set = np.array([])
    exp1_set = np.array([])
    exp2_set = np.array([])
    exp3_set = np.array([])
    molecular_datafile = open(csv_filepath, "r")
    features = pd.read_csv(csv_filepath)
    columns = features.columns
    placeholder_column = columns[0]
    del placeholder_column
    r_vals = features["r"]
    ACSF_vals = features["ACSF"]
    z_vals = features["z"]
    d1_vals = features["d1"]
    d2_vals = features["d2"]
    d3_vals = features["d3"]
    exp1_vals = features["exp1"]
    exp2_vals = features["exp2"]
    exp3_vals = features["exp3"]
    for i in r_vals:
        r_set = np.append(r_set, i)
    for j in ACSF_vals:
        ACSF_set = np.append(ACSF_set, j)
    for k in z_vals:
        z_set = np.append(z_set, k) 
    for l in d1_vals:
        d1_set = np.append(d1_set, l)
    for m in d2_vals:
        d2_set = np.append(d2_set, m)
    for n in d3_vals:
        d3_set = np.append(d3_set, n)
    for o in exp1_vals:
        exp1_set = np.append(exp1_set, o)
    for p in exp2_vals:
        exp2_set = np.append(exp2_set, p)
    for q in exp3_vals:
        exp3_set = np.append(exp3_set, q)
    record_keys = ["r", "ACSF", "z", "d1", "d2", "d3", "exp1", "exp2", "exp3"]
    num_records = np.arange(0, len(record_keys))
    for i in num_records:
        if record_keys[i] == "r":
            records_dict.update({record_keys[i]:list(r_set)})
        elif record_keys[i] == "ACSF":
            records_dict.update({record_keys[i]:list(ACSF_set)})
        elif record_keys[i] == "z":
            records_dict.update({record_keys[i]:list(z_set)})
        elif record_keys[i] == "d1":
            records_dict.update({record_keys[i]:list(d1_set)})
        elif record_keys[i] == "d2":
            records_dict.update({record_keys[i]:list(d2_set)})
        elif record_keys[i] == "d3":
            records_dict.update({record_keys[i]:list(d3_set)})
        elif record_keys[i] == "exp1":
            records_dict.update({record_keys[i]:list(exp1_set)})
        elif record_keys[i] == "exp2":
            records_dict.update({record_keys[i]:list(exp2_set)})
        elif record_keys[i] == "exp3":
            records_dict.update({record_keys[i]:list(exp3_set)})
        i += 1        
    return records_dict

def take_data_batch(csv_filepath):
    data_dict = initialize_dataset(csv_filepath)
    example_set = tf.train.Example(features = tf.train.Features(feature={
    'r': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data_dict["r"][0]])),
    'ACSF': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data_dict["ACSF"][0]])),
    'z': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data_dict['z'][0]])),
    'd1': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data_dict['d1'][0]])),
    'd2': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data_dict['d2'][0]])),
    'd3': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data_dict['d3'][0]])),
    'exp1': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data_dict['exp1'][0]])),
    'exp2': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data_dict['exp2'][0]])),
    'exp3': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data_dict['exp3'][0]]))
    }))
    return example_set

def write_TF_records_file(csv_filepath):
    encoded_dataset = take_data_batch(csv_filepath)
    with tf.io.TFRecordWriter("test_data.tfrecord") as writer:
        writer.write(encoded_dataset.SerializeToString())

write_TF_records_file(csv_filepath)
record_floats = take_data_batch(csv_filepath)

def nodeset_position_fxn(nodeset_proto):
    feature_descriptor = {"z" : tf.io.FixedLenFeature([], tf.float32), "r" : tf.io.FixedLenFeature([], tf.float32), "ACSF" : tf.io.FixedLenFeature([], tf.float32), "d1" : tf.io.FixedLenFeature([], tf.float32), "d2" : tf.io.FixedLenFeature([], tf.float32), "d3" : tf.io.FixedLenFeature([], tf.float32), "exp1" : tf.io.FixedLenFeature([], tf.float32), "exp2" : tf.io.FixedLenFeature([], tf.float32), "exp3" : tf.io.FixedLenFeature([], tf.float32)}
    position_fxn = tf.io.parse_single_example(nodeset_proto, feature_descriptor)
    return position_fxn

csv_filepath = "/users/haydenprescott/documents/test.csv"
def record_floats(value):
    r_floats = tf.train.Feature(float_list = tf.train.FloatList(value = [value]))
    return r_floats

def serialize_floats(r_val, ACSF_val, z_val, d1_val, d2_val, d3_val, exp1_val, exp2_val, exp3_val, csv_filepath):
    initial_state_feature = {"r":record_floats(r_val), "ACSF":record_floats(ACSF_val), "z":record_floats(z_val), "d1":record_floats(d1_val), "d2":record_floats(d2_val), "d3":record_floats(d3_val), "exp1":record_floats(exp1_val), "exp2":record_floats(exp2_val), "exp3":record_floats(exp3_val)}
    initial_state_proto = tf.train.Example(features = tf.train.Features(feature = initial_state_feature))
    initial_state_records = initial_state_proto.SerializeToString()
    return initial_state_records

def convert_floats_to_records(csv_filepath, record_filepath, record_filename):
    data_dict = initialize_dataset(csv_filepath)
    previous_record_path = Path(record_filename)
    serialized_input_floats = []
    n_input_vals = np.arange(0, (len(data_dict['r'])))
    i = 0
    for i in n_input_vals:
        r, ACSF, z, d1, d2, d3, exp1, exp2, exp3 = data_dict['r'][i], data_dict['ACSF'][i], data_dict['z'][i], data_dict['d1'][i], data_dict['d2'][i], data_dict['d3'][i], data_dict['exp1'][i], data_dict['exp2'][i], data_dict['exp3'][i]
        all_floats_per_molecule = (r, ACSF, z, d1, d2, d3, exp1, exp2, exp3)
        serialized_input_floats.append(all_floats_per_molecule)
        initial_state_data = tf.constant([['r', 'ACSF', 'z', 'd1', 'd2', 'd3','exp1', 'exp2', 'exp3']])
    with tf.io.TFRecordWriter(record_filepath) as record_writer:
        for r, ACSF, z, d1, d2, d3, exp1, exp2, exp3 in serialized_input_floats:
            serialized_input_floats = serialize_floats(r, ACSF, z, d1, d2, d3, exp1, exp2, exp3, csv_filepath)
            record_writer.write(serialized_input_floats)
            record_batch = tf.data.TFRecordDataset(record_filename)
            record_batch = record_batch.map(nodeset_position_fxn)
            ith_atom_features = tf.constant([[str(r), str(ACSF), str(z), str(d1), str(d2), str(d3), str(exp1), str(exp2), str(exp3)]])
            initial_state_data = tf.concat([initial_state_data, ith_atom_features], 0)
    return initial_state_data


def build_graph_tensor(csv_filepath, record_filepath, record_filename):
    initial_graph_tensor = convert_floats_to_records(csv_filepath, record_filepath, record_filename)
    input_graph_tensor = tf.transpose(initial_graph_tensor)
    input_rvals = input_graph_tensor[0][1:len(input_graph_tensor) - 1].numpy()
    input_ACSFvals = input_graph_tensor[1][1:len(input_graph_tensor) - 1].numpy()
    input_zvals = input_graph_tensor[2][1:len(input_graph_tensor) - 1].numpy()
    input_d1vals = input_graph_tensor[3][1:len(input_graph_tensor) - 1].numpy()
    input_d2vals = input_graph_tensor[4][1:len(input_graph_tensor) - 1].numpy()
    input_d3vals = input_graph_tensor[5][1:len(input_graph_tensor) - 1].numpy()
    input_exp1vals = input_graph_tensor[6][1:len(input_graph_tensor) - 1].numpy()
    input_exp2vals = input_graph_tensor[7][1:len(input_graph_tensor) - 1].numpy()
    input_exp3vals = input_graph_tensor[8][1:len(input_graph_tensor) - 1].numpy()
    input_graph_tensor = np.array([input_rvals, input_ACSFvals, input_zvals, input_d1vals, input_d2vals, input_d3vals, input_exp1vals, input_exp2vals, input_exp3vals])
    input_graph_tensor = input_graph_tensor.transpose()
    initial_graph_tensor = initial_graph_tensor.numpy()
    initial_state_and_GNN_input = (initial_graph_tensor, input_graph_tensor)
    return initial_state_and_GNN_input

graph_tensor = build_graph_tensor(csv_filepath, record_filepath, record_filename)

class find_orbitals_get_characteristics:
    def __init__(self, molecular_formula, seed_structure_directory, element, graph_tensor):
        self.molecular_formula = molecular_formula
        self.seed_structure_directory = seed_structure_directory
        self.element = element
        self.graph_tensor = graph_tensor

    def find_atomic_symbol(self, element):
        characteristic_list = []
        blanks = np.array([])
        for i in np.arange(0, len(str(element))):
            characteristic_list.append(str(element)[i])
        for j in np.arange(0, len(characteristic_list)):
            if characteristic_list[j] == " ":
                blanks = np.append(blanks, j)
        atomic_symbol = str(element)[int(blanks[0]) + 1:int(blanks[1])]
        return atomic_symbol

    def find_atomic_number(self, element):
        characteristic_list = []
        blanks = np.array([])
        for i in np.arange(0, len(str(element))):
            characteristic_list.append(str(element)[i])
        for j in np.arange(0, len(characteristic_list)):
            if characteristic_list[j] == " ":
                blanks = np.append(blanks, j)
        atomic_symbol = str(element)[0:int(blanks[0])]
        return atomic_symbol

    
    def get_symbols_and_numbers(self, molecular_formula, seed_structure_directory, graph_tensor):
        seed_structure = seed_structure_directory + "/" + str(molecular_formula) + ".csv"
        molecular_details = open(seed_structure, "r")
        molecular_details = pd.read_csv(seed_structure)
        element_symbols = np.array(molecular_details['symbol'][0:len(graph_tensor[0]) - 1])
        atomic_numbers = np.array([])
        valence_blocks = np.array([])
        element_set = ptable.get_all_elements()
        for j in np.arange(0, len(element_symbols)):
            for i in np.arange(0, len(element_set)):
                if str.encode(self.find_atomic_symbol(element_set[i])) == str.encode(element_symbols[j]):
                    atomic_numbers = np.append(atomic_numbers, int(self.find_atomic_number(element_set[i])))
                    valence_blocks = np.append(valence_blocks, element_set[i].block)
        return element_symbols, atomic_numbers, valence_blocks


import pandas as pd
import os as os
import mendeleev as ptable
import numpy as np
from Determine_Bonds_Get_Input import find_orbitals_get_characteristics, build_graph_tensor

class g16_input_file:
    def __init__(self, seed_structure_directory, molecular_formula, job_type, input_directory, output_directory, AO_basis, polarization, job_title, SCF_optimization, csv_filepath, record_filepath, record_filename, element, n_CPU, allowed_memory):
        self.seed_structure_directory = seed_structure_directory
        self.molecular_formula = molecular_formula
        self.job_type = job_type
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.AO_basis = AO_basis
        self.polarization = polarization
        self.job_title = job_title
        self.SCF_optimization = SCF_optimization
        self.csv_filepath = csv_filepath
        self.record_filepath = record_filepath
        self.record_filename = record_filename
        self.element = element
        self.n_CPU = n_CPU
        self.allowed_memory = allowed_memory

    def generate_input_file(self, molecular_formula, job_type, input_directory, output_directory, seed_structure_directory, csv_filepath, record_filepath, record_filename, n_CPU, allowed_memory):
        input_filepath = os.path.join(input_directory, molecular_formula)
        input_file = open(input_filepath, "w")
        seed_structure = seed_structure_directory + "/" + molecular_formula + ".csv"
        energy_checkpoint_path = os.path.join(output_directory, molecular_formula)
        geom_checkpoint_path = os.path.join(output_directory, molecular_formula + "_geom")
        energy_chk = open(energy_checkpoint_path, "w")
        geom_chk = open(geom_checkpoint_path, "w")
        energy_chk_name = str(molecular_formula) + ".chk"
        geom_chk_name = str(molecular_formula) + "_geom" + ".chk"
        if job_type == "ground_state_energy_calculation":
            job_abbrev = "SP"
            input_file.write("%chk=" + energy_chk_name)
            input_file.write("\n")
        elif job_type == "geometry_optimization":
            job_abbrev = "Opt"
            input_file.write("%chk=" + geom_chk_name)
            input_file.write("\n")
        graph_tensor = build_graph_tensor(csv_filepath, record_filepath, record_filename)
        molecular_characteristics = find_orbitals_get_characteristics(molecular_formula = molecular_formula, seed_structure_directory = seed_structure_directory, element = None, graph_tensor = graph_tensor)
        symbol, Z, block = molecular_characteristics.get_symbols_and_numbers(molecular_formula = molecular_characteristics.molecular_formula, seed_structure_directory = molecular_characteristics.seed_structure_directory, graph_tensor = molecular_characteristics.graph_tensor)
        if "d" in block or "f" in block:
            AO_basis = "triple_zeta"
        elif "C" in symbol:
            AO_basis = "6-31G"
        elif "Na" in symbol or "K" in symbol or "Mg" in symbol or "Ca" in symbol or "Li" in symbol or "Be" in symbol:
            AO_basis = "LANL_double_zeta"
        else:
            AO_basis = "STO-3G"
        input_file.write("%nprocs=" + str(n_CPU))
        input_file.write("\n")
        input_file.write("%mem=" + allowed_memory)
        input_file.write("\n")
        if AO_basis == "triple_zeta":
            if polarization == True:
                input_file.write("# 3-21G**" + " " + job_abbrev + " " + SCF_optimization)
                input_file.write("\n")
            else:
                input_file.write("# 3-21G" + " " + job_abbrev + " " + SCF_optimization)
                input_file.write("\n")
        elif AO_basis == "6-31G":
            if polarization == True:
                input_file.write("# 6-31G**" + " " + job_abbrev + " " + SCF_optimization)
                input_file.write("\n")
            else: 
                input_file.write("# " + AO_basis + " " + job_abbrev + " " + SCF_optimization)
                input_file.write("\n")
        elif AO_basis == "LANL_double_zeta":
            input_file.write("LANL2DZ" + " " + job_abbrev + " " + SCF_optimization)
            input_file.write("\n")
        input_file.write(job_title)
        input_file.write("\n")
        input_file.write(str(0) + "  " + str(1))
        atomic_pos = pd.read_csv(seed_structure)
        labels = atomic_pos.insert(0, "labels", symbol, True)
        atomic_pos = atomic_pos.to_numpy()
        atomic_pos = atomic_pos[1:]
        labels = labels[1:].to_numpy()
        for i in np.arange(0, len(atomic_pos.transpose())):
            coordinate_specification = " " + str(labels.transpose()[0][i])
            for j in np.arange(0, len(atomic_pos.transpose()[i])):
                coordinate_specification = coordinate_specification + " " + str(atomic_pos[i][j])
            input_file.write(coordinate_specification)
            input_file.write("\n")
        return input_filepath
        
        
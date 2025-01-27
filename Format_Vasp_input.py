def find_atomic_symbol(element):
    characteristic_list = []
    blanks = np.array([])
    for i in np.arange(0, len(str(element))):
        characteristic_list.append(str(element)[i])
    for j in np.arange(0, len(characteristic_list)):
        if characteristic_list[j] == " ":
            blanks = np.append(blanks, j)
    atomic_symbol = str(element)[int(blanks[0]):int(blanks[1])]
    return atomic_symbol

atomic_numbers = np.array([])
element_set = ptable.get_all_elements()
for i in np.arange(0, len(element_set)):
    if find_atomic_symbol(element_set[i]) == find_atomic_symbol(element_set[56]):
        print('lanthanum')
        print(element_set[i].block)


class VASP_input:
    def __init__(self, working_directory, molecular_formula, job_type, seed_structure_directory, csv_filepath, record_filepath, record_filename, pseudopotential, xc_functional, cluster_type, principal_lattice_vector, accuracy_bound, max_iterations, complex_rotation_term, max_iterations):
        self.working_directory = working_directory
        self.molecular_formula = molecular_formula
        self.job_type = job_type
        self.seed_structure_directory = seed_structure_directory
        self.csv_filepath = csv_filepath
        self.record_filepath = record_filepath
        self.record_filename = record_filename
        self.pseudopotential = pseudopotential
        self.xc_functional = xc_functional
        self.cluster_type = cluster_type
        self.principal_lattice_vector = principal_lattice_vector
        self.accuracy_bound = accuracy_bound
        self.max_iterations = max_iterations
        self.complex_rotation_term = complex_rotation_term
        self.rotation_angle = rotation_angle

    def convert_rotation_to_matrix(self, complex_rotation_term, rotation_angle):
    rotation_str_representation = str(complex_rotation_term)
    if rotation_str_representation[0] == '[':
        rotation_obj = scipy.spatial.transform.Rotation.from_quat(complex_rotation_term)
        rotation_matrix = rotation_term.as_matrix()
    else:
        rotation_matrix = np.array([[sympy.cos(rotation_angle), -sympy.sin(rotation_angle)], [sympy.sin(rotation_angle), sympy.cos(rotation_angle)]])
    return rotation_matrix
    
    def find_mapping_to_coordinate_axis(self, principal_lattice_vector, accuracy_bound, max_iterations): 
        orthogonal_shift = principal_lattice_vector[2]
        rotation_steps = []
        starting_vector = principal_lattice_vector
        update_index = 0
        n_transformations = 0
        while n_transformations < max_iterations:
            lattice_norm = np.sqrt(principal_lattice_vector[0]**2 + principal_lattice_vector[1]**2 + principal_lattice_vector[1]**2)
            xy_trace = np.array([principal_lattice_vector[0], principal_lattice_vector[1], 0])
            trace_norm = np.sqrt(xy_trace[0]**2 + xy_trace[1]**2 + xy_trace[2]**2)
            xy_trace = np.array([xy_trace[0] * (lattice_norm / trace_norm), xy_trace[1] * (lattice_norm / trace_norm), xy_trace[2] * (lattice_norm / trace_norm)])
            a = sympy.symbols('a', real = True)
            solution_space = sympy.solve(lattice_norm**2 * sympy.cos(a) - np.dot(principal_lattice_vector, xy_trace), a)
            if len(solution_space) > 0:
                angular_shift = (2 * np.pi) - solution_space[0] 
                axis_of_rotation = np.cross(principal_lattice_vector, xy_trace)
                rotation_quat = np.array([sympy.cos(angular_shift / 2), axis_of_rotation[0] * sympy.sin(angular_shift / 2), axis_of_rotation[1] * sympy.sin(angular_shift / 2), axis_of_rotation[2] * sympy.sin(angular_shift / 2)])
                rotation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = rotation_quat, rotation_angle = angular_shift)
                checkpoint = principal_lattice_vector
                checkpoint_matrix = rotation_matrix
                principal_lattice_vector = principal_lattice_vector @ rotation_matrix
                new_shift = principal_lattice_vector[2]
                if abs(new_shift) < abs(orthogonal_shift):
                    rotation_steps.append(rotation_matrix)
                    n_transformations += 1
                    orthogonal_shift = new_shift
                else:
                    angular_shift = -1 * ((2 * np.pi) - solution_space[0])
                    axis_of_rotation = np.cross(checkpoint, xy_trace)
                    rotation_quat = np.array([sympy.cos(angular_shift / 2), axis_of_rotation[0] * sympy.sin(angular_shift / 2), axis_of_rotation[1] * sympy.sin(angular_shift / 2), axis_of_rotation[2] * sympy.sin(angular_shift / 2)])
                    rotation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = rotation_quat, rotation_angle = angular_shift)
                    principal_lattice_vector = checkpoint @ rotation_matrix
                    new_shift = principal_lattice_vector[2]
                    n_transformations += 1
                    if abs(new_shift) < abs(orthogonal_shift):
                        rotation_steps.append(rotation_matrix)
                        n_transformations += 1
                        orthogonal_shift = new_shift
                    else:
                        if orthogonal_shift > 0:
                            variation_angle = -1 * (np.pi / 180)
                        else:
                            variation_angle = np.pi / 180
                        axis_of_rotation = np.cross(checkpoint, xy_trace)
                        checkpoint_norm = sympy.sqrt(checkpoint[0]**2 + checkpoint[1]**2 + checkpoint[2]**2)
                        variation_quat = np.array([sympy.cos(variation_angle / 2), axis_of_rotation[0] * sympy.sin(variation_angle / 2), axis_of_rotation[1] * sympy.sin(variation_angle / 2), axis_of_rotation[2] * sympy.sin(variation_angle / 2)])
                        variation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = variation_quat, rotation_angle = variation_angle)
                        varied_components = checkpoint @ variation_matrix
                        rotation_steps.append(variation_matrix)
                        principal_lattice_vector = varied_components
                        orthogonal_shift = principal_lattice_vector[2]
                        n_transformations += 1
            if abs(orthogonal_shift) >= abs(accuracy_bound):
               continue
            else:
                break  
            
        flattening_rotation = np.eye(len(rotation_steps[0]))
        for i in np.arange(0, len(rotation_steps)):
            flattening_rotation = flattening_rotation @ rotation_steps[i]
        u, s, v = scipy.linalg.svd(flattening_rotation)
        reciporical_singular_vals = np.eye(len(s))
        for i in np.arange(0, len(s)):
            for j in np.arange(0, len(s)):
                if reciporical_singular_vals[i][j] == 1:
                    reciporical_singular_vals[i][j] = 1 / s[j]
        u_inv = np.linalg.inv(u)
        v_inv_t = np.linalg.inv(v).transpose()
        reverse_flattening_rotation = u_inv @ reciporical_singular_vals @ v_inv_t

        prev_lattice_vector = principal_lattice_vector
        principal_lattice_vector = np.array([prev_lattice_vector[0], prev_lattice_vector[1]])
        planar_shift = principal_lattice_vector[1]
        rotation_steps = []
        update_index = 0
        planar_transformations = 0
        while planar_transformations < max_iterations:
            planar_norm = np.sqrt(principal_lattice_vector[0]**2 + principal_lattice_vector[1]**2)
            unit_cell_axis = np.array([principal_lattice_vector[0], 0])
            cell_axis_norm = np.sqrt(unit_cell_axis[0]**2 + unit_cell_axis[1]**2)
            unit_cell_axis = np.array([unit_cell_axis[0] * (planar_norm / cell_axis_norm), unit_cell_axis[1] * (planar_norm / cell_axis_norm)])
            a = sympy.symbols('a', real = True)
            solution_space = sympy.solve(planar_norm**2 * sympy.cos(a) - np.dot(principal_lattice_vector, unit_cell_axis), a)
            if len(solution_space) > 0:
                xy_angular_shift = (2 * np.pi) - solution_space[0] 
                axis_of_rotation = np.array([0,0,1])
                imag_term = sympy.I
                rotation_number = sympy.cos(xy_angular_shift) + sympy.sin(xy_angular_shift) * imag_term
                rotation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = rotation_number, rotation_angle = xy_angular_shift)
                checkpoint = principal_lattice_vector
                checkpoint_matrix = rotation_matrix
                principal_lattice_vector = principal_lattice_vector @ rotation_matrix
                new_shift = principal_lattice_vector[1]
                if abs(new_shift) < abs(planar_shift):
                    rotation_steps.append(rotation_matrix)
                    planar_transformations += 1
                    planar_shift = new_shift
                else:
                    xy_angular_shift = -1 * ((2 * np.pi) - solution_space[0])
                    axis_of_rotation = np.array([0,0,1])
                    rotation_number = sympy.cos(xy_angular_shift) + sympy.sin(xy_angular_shift) * imag_term
                    rotation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = rotation_number, rotation_angle = xy_angular_shift)
                    principal_lattice_vector = checkpoint @ rotation_matrix
                    new_shift = principal_lattice_vector[1]
                    n_transformations += 1
                    if abs(new_shift) < abs(planar_shift):
                        rotation_steps.append(rotation_matrix)
                        planar_transformations += 1
                        planar_shift = new_shift
                    else:
                        if planar_shift > 0:
                            variation_angle = -1 * (np.pi / 180)
                        else:
                            variation_angle = np.pi / 180
                        axis_of_rotation = np.array([0,0,1])
                        checkpoint_norm = sympy.sqrt(checkpoint[0]**2 + checkpoint[1]**2)
                        variation_number = np.cos(variation_angle) + np.sin(variation_angle) * imag_term
                        variation_matrix = self.convert_rotation_to_matrix(complex_rotation_term = variation_number, rotation_angle = variation_angle)
                        varied_components = checkpoint @ variation_matrix
                        rotation_steps.append(variation_matrix)
                        principal_lattice_vector = varied_components
                        planar_shift = principal_lattice_vector[1]
                        planar_transformations += 1
            if abs(planar_shift) >= abs(accuracy_bound):
                continue
            else:
                break  

        unit_cell_axis = principal_lattice_vector
        axis_alignment = sympy.eye(len(rotation_steps[0]))
        for i in np.arange(0, len(rotation_steps)):
            axis_alignment = axis_alignment @ sympy.Matrix(rotation_steps[i])
        u, s, v = axis_alignment.singular_value_decomposition()
        s = np.diag(np.array(s))
        reciporical_singular_vals = np.eye(len(s))
        for i in np.arange(0, len(s)):
            for j in np.arange(0, len(s)):
               if reciporical_singular_vals[i][j] == 1:
                    reciporical_singular_vals[i][j] = 1 / s[j]
        reciporical_singular_vals = sympy.Matrix(reciporical_singular_vals)
        u_inv = u.inv()
        v_inv_t = u.inv().transpose()
        reverse_axis_alignment = u_inv @ reciporical_singular_vals @ v_inv_t
        initial_principal_norm = np.sqrt(starting_vector[0]**2 + starting_vector[1]**2 + starting_vector[2]**2)
        aligned_principal_norm = np.sqrt(principal_lattice_vector[0]**2 + principal_lattice_vector[1]**2 + principal_lattice_vector[2]**2)
        #principal_lattice_vector = np.array([unit_cell_axis[0], unit_cell_axis[1], prev_lattice_vector[2]])
        #principal_lattice_vector = np.array([principal_lattice_vector[0] * (initial_principal_norm / aligned_principal_norm), principal_lattice_vector[1] * (initial_principal_norm / aligned_principal_norm), principal_lattice_vector[2] * (initial_principal_norm / aligned_principal_norm)])
        return flattening_rotation, reverse_flattening_rotation, axis_alignment, reverse_axis_alignment, initial_principal_norm, aligned_principal_norm

   
    def generate_cluster_object(self, seed_structure_directory, molecular_formula, cluster_type, csv_filepath, record_filepath, record_filename, accuracy_bound, max_iterations):
        seed_structure = seed_structure_directory + "/" + molecular_formula + ".csv"
        graph_tensor = build_graph_tensor(csv_filepath, record_filepath, record_filename)
        molecular_characteristics = find_orbitals_get_characteristics(molecular_formula = molecular_formula, seed_structure_directory = seed_structure_directory, element = None, graph_tensor = graph_tensor)
        symbol, Z, block = molecular_characteristics.get_symbols_and_numbers(molecular_formula = molecular_characteristics.molecular_formula, seed_structure_directory = molecular_characteristics.seed_structure_directory, graph_tensor = molecular_characteristics.graph_tensor)
        species_in_molecule = []
        coordinate_vectors = []
        if cluster_type == "molecule":
            for i in np.arange(0, len(symbol)):
                species_in_molecule.append(str(symbol[i]))
            atomic_pos_components = pd.read_csv(seed_structure)
            atomic_pos_components = atomic_pos_components[:len(graph_tensor[1])]
            atomic_pos_components = atomic_pos_components.to_numpy()
            atomic_pos_components = atomic_pos_components.transpose()[0:4]
            atomic_pos_components = atomic_pos_components.transpose()
            for j in np.arange(0, len(atomic_pos_components)):
                coordinate_vectors.append(atomic_pos_components[j][1:])
            molecule = pmtg.core.structure.Molecule(species_in_molecule, coordinate_vectors)
            cluster = molecule
        elif cluster_type == "periodic":
            bravais_lattice = pd.read_csv(seed_structure)
            bravais_lattice = bravais_lattice.to_numpy()
            representative_unit_cell = random.randint(0, len(bravais_lattice))
            cell_norms = np.array([])
            representative_norms = np.array([])
            species_index = 0
            for i in np.arange(0, len(bravais_lattice[representative_unit_cell + species_index])):
                for j in np.arange(0, len(bravais_lattice)):
                    for k in np.arange(0, len(bravais_lattice[j])):

                        representative_norms = np.append(representative_norms, np.sqrt(np.sum(abs(bravais_lattice[representative_unit_cell + species_index][i]**2 - bravais_lattice[j][k]**2))))
            i = 1
            refernece_unit_cell_norm = representative_norms[0]
            for i in np.arange(1, len(representative_norms)):
                if representative_norms[i] == reference_unit_cell_norm:
                    reference_list = list(representative_norms)
                    unit_cell_size = refernece_list.index(representative_norms[i]) - reference_list.index(reference_unit_cell_norm)
                    break
            guess_cell = representative_norms[0:unit_cell_size - 1]
            for j in np.arange(0, len(guess_cell)):
                if guess_cell[j] == representative_norms[unit_cell_size + j]:
                    cell_norms = np.append(cell_norms, guess_cell)
            cell_coordinates = bravais_lattice.transpose()[0:len(cell_norms) - 1]
            cell_coordinates = cell_coordinates.transpose()
            principal_cell_dimension = max(cell_norms)
            principal_lattice_vector = np.array([cell_coordinates[list(cell_norms).index(principal_cell_dimension)][0], cell_coordinates[list(cell_norms).index(principal_cell_dimension)][1], cell_coordinates[list(cell_norms).index(principal_cell_dimension)][2]])
            xy_plane_rotation, xy_plane_reverse, coordinate_axis_alignment, coordinate_axis_reverse, lattice_vector_norm, aligned_vector_norm = self.find_mapping_to_coordinate_axis(principal_lattice_vector, accuracy_bound, max_iterations)
            principal_vector_planar = principal_lattice_vector @ xy_plane_rotation
            principal_lattice_vector = np.array([principal_vector_planar[0], principal_vector_planar[1]])
            principal_vector_aligned = principal_lattice_vector @ coordinate_axis_alignment
            principal_lattice_vector = np.array([principal_vector_aligned[0], principal_vector_aligned[1], principal_vector_planar[2]])
            principal_lattice_vector = np.array([principal_lattice_vector[0] * (lattice_vector_norm / aligned_vector_norm), principal_lattice_vector[1] * (lattice_vector_norm / aligned_vector_norm), principal_lattice_vector[2] * (lattice_vector_norm / aligned_vector_norm)])
            aligned_unit_cell = []
            for k in range(0, len(cell_norms)):
                lattice_vector = np.array([cell_coordinates[k][0], cell_coordinates[k][1], cell_coordinates[k][2]])
                norm = np.sqrt(lattice_vector[0]**2 + lattice_vector[1]**2 + lattice_vector[2]**2)
                planar = lattice_vector @ xy_plane_rotation
                lattice_vector = np.array([planar[0], planar[1]])
                aligned = lattice_vector @ coordinate_axis_alignment
                lattice_vector = np.array([aligned[0], aligned[1], planar[2]])
                aligned_norm = np.sqrt(lattice_vector[0]**2 + lattice_vector[1]**2 + lattice_vector[2]**2)
                lattice_vector = np.array([lattice_vector[0] * (norm / aligned_norm), lattice_vector[1] * (norm / aligned_norm), lattice_vector[2] * (norm / aligned_norm)])
                aligned_unit_cell.append(lattice_vector)
        potcar = pmtg.io.vasp.Potcar([symbol], functional = xc_functional)
        return aligned_unit_cell
                    

    
                 
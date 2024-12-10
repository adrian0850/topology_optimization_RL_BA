import numpy as np
import constants as const

def scale_matrix(matrix, target_rows, target_cols):
    """A function that scales a given matrix to a target size. 
    The scaling is done by repeating the elements of the matrix.
    ----------
    Parameters:\n
    - matrix : numpy.ndarray
        - The matrix that should be scaled.
    - target_rows : int
        - The target number of rows of the scaled matrix.
    - target_cols : int
        - The target number of columns of the scaled matrix.
    -------
    Returns:\n
    - numpy.ndarray
        - The scaled matrix.
    """
    original_rows = len(matrix)
    original_cols = len(matrix[0]) if original_rows > 0 else 0
    
    # Calculate scale factors
    row_scale_factor = target_rows // original_rows
    col_scale_factor = target_cols // original_cols
    
    # Scale the matrix
    scaled_matrix = []
    for row in matrix:
        # Scale each row horizontally
        scaled_row = []
        for element in row:
            scaled_row.extend([element] * col_scale_factor)
        
        # Scale the matrix vertically
        for _ in range(row_scale_factor):
            scaled_matrix.append(list(scaled_row))
    
    # Handle any remaining rows due to non-integer scale factors
    additional_rows = target_rows % original_rows
    if additional_rows > 0:
        for i in range(additional_rows):
            scaled_matrix.append(list(scaled_matrix[i]))
    
    # Handle any remaining columns due to non-integer scale factors
    additional_cols = target_cols % original_cols
    if additional_cols > 0:
        for row in scaled_matrix:
            row.extend(row[:additional_cols])
    
    return np.array(scaled_matrix)



def extract_fem_data(design_matrix):
    nodes, elements, node_map = extract_nodes_and_elements(design_matrix)
    boundary_conditions = extract_boundary_conditions(design_matrix, node_map)
    loads = extract_loads(design_matrix, node_map)
    return nodes, elements, boundary_conditions, loads

def extract_nodes_and_elements(design):
    nodes = []
    elements = []
    node_id = 0
    node_map = {}

    existing_elements = set()  # Store unique element combinations


    height, width = design.shape[1], design.shape[2]

    for i in range(height):
        for j in range(width):
            # Define nodes for the pixel
            pixel_nodes = [
                (i, j), (i, j + 1),
                (i + 1, j), (i + 1, j + 1)
            ]

            for node in pixel_nodes:
                if node not in node_map:
                    node_map[node] = node_id
                    nodes.append([float(node[0]), float(node[1])])
                    node_id += 1

            # Define elements by connecting nodes as two triangles
            n1, n2, n3, n4 = (node_map[(i, j)], node_map[(i, j + 1)],
                            node_map[(i + 1, j)], node_map[(i + 1, j + 1)])

            # Check if the element or its reversed version already exists
            element_tuple = tuple(sorted([n1, n2, n3, n4]))
            if element_tuple not in existing_elements:
                # Create two triangles with a flag indicating if they are voided
                is_voided = design[0, i, j] == 0
                elements.append([n1, n3, n4, n1, is_voided])  # Triangle 1
                elements.append([n1, n4, n2, n1, is_voided])  # Triangle 2
                existing_elements.add(element_tuple)

    return nodes, elements, node_map

def extract_boundary_conditions(design_matrix, node_map):
    boundary_conditions = []
    rows, cols = design_matrix[1,:,:].shape

    for i in range(rows):
        for j in range(cols):
            if design_matrix[1, i, j] == 1:
                # Apply boundary conditions to all corners of the pixel
                pixel_nodes = [
                    (i, j), (i, j+1),
                    (i+1, j), (i+1, j+1)
                ]
                for node in pixel_nodes:
                    node_id = node_map.get(node)
                    if node_id is not None:
                        boundary_conditions.append([node_id, 1, 1, 0.0])  # Example: fixed in x-direction
                        boundary_conditions.append([node_id, 2, 2, 0.0])  # Example: fixed in y-direction
    return boundary_conditions

def extract_loads(design_matrix, node_map):

    loads = []
    rows, cols = design_matrix[2,:,:].shape
    
    for i in range(rows):
        for j in range(cols):
            if design_matrix[2, i, j] != 0 or design_matrix[3, i, j] != 0:
                # Apply loads to all corners of the pixel
                pixel_nodes = [
                    (i, j), (i, j+1),
                    (i+1, j), (i+1, j+1)
                ]
                for node in pixel_nodes:
                    node_id = node_map.get(node)
                    if node_id is not None:
                        if design_matrix[2, i, j] != 0:
                            loads.append([node_id, 1, design_matrix[2, i, j]])
                        if design_matrix[3, i, j] != 0:
                            loads.append([node_id, 2, design_matrix[3, i, j]])
    if not loads:
        print("design_matrix", design_matrix)
        raise ValueError("No loads found in the design matrix.")
    return loads

def encode_loaded_nodes(grid, coordinate_list):
    for (row, col, val) in coordinate_list:
        if val[1] == "Y":
            grid[2][row][col] = int(val[2:])
        elif val[1] == "X":
            grid[3][row][col] = int(val[2:])
    return grid

def encode_bounded_elements(grid, coordinate_list):
    for (row, col) in coordinate_list:
        grid[1][row][col] = 1
    return grid

def create_grid(height, width, bounded, loaded):
    grid = np.zeros((4, height, width), dtype=np.float32)
    grid[0, :, :] = 1
    grid = encode_bounded_elements(grid, bounded)
    grid = encode_loaded_nodes(grid, loaded)
    #print(grid)
    return grid

def remove_material(grid, row, col):
    grid[0,row,col] = 0
    return grid

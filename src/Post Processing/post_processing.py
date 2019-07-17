from pyevtk.hl import gridToVTK
import numpy as np
import csv
import array
import var
import math
import pprint

# Return line or column of the matrix, given the axis and the index
# \param matr (list 2D): matrix 2d
# \param axis (char): axis of the values ('x' or 'i' to get column and 'y' or 'j' to get line)
# \param index (int): index of the axis
# \return (list 1D) line or column from matrix
def line_from_matrix(matr, axis, index):
    if (axis=='x' or axis=='i'):
        return np.array(matr[:,index])
    elif (axis=='y' or axis=='j'):
        return np.array(matr[index,:])

# Return matrix2d from a 3d, given the axis and the index to be constant
# \param matr (list 3D): matrix 3d [x][y][z]
# \param axis ((char,char)): tuple with axis of the values ('x' or 'i': column, 'y' or 'j': row, 'z' or 'k': z axis)
# \param index ((int,int)): tuple with index of the axis
# \return (list 2D) 2d matrix
def line_from_matrix3d(matr, axis, index):
    if (axis[0]=='x' or axis[0]=='i'):
        if(axis[1]=='y' or axis[1]=='j'):
            return np.array(matr[index[0],index[1],:])
        if(axis[1]=='z' or axis[1]=='k'):
            return np.array(matr[index[0],:,index[1]])
    elif (axis[0]=='y' or axis[0]=='j'):
        if(axis[1]=='x' or axis[1]=='i'):
            return np.array(matr[index[0],index[1],:])
        if(axis[1]=='z' or axis[1]=='k'):
            return np.array(matr[:,index[0],index[1]])
    elif (axis[0]=='z' or axis[0]=='k'):
        if(axis[1]=='x' or axis[1]=='i'):
            return np.array(matr[index[1],:,index[1]])
        if(axis[1]=='y' or axis[1]=='j'):
            return np.array(matr[:,index[1],index[0]])


# Normalize the array by U_MAX
# \param vec (list): array to be normalized
# \return (list) normalized array
def normalize_array(vec):
    return (np.divide(vec, var.U_MAX))


# Flip matrix vertically
# \param matr (list 2D): matrix to be flipped
# \return (list 2D) flipped matrix
def flip_matr_ver(matr):
    return np.flipud(matr)


# Reshapes a 1D array to a 2D matrix
# \param vec (list 1D): 1D array to be reshaped
# \return (list 2D) reshaped 2D matrix
def array2matrix(vec):
    return np.reshape(vec, (var.NY, var.NX), 'C') #[y][x]

# Reshapes a 1D array to a 3D matrix
# \param vec (list 1D): 1D array to be reshaped
# \return (np.array 3D) reshaped 3D matrix
def array2matrix3d(vec):
    return np.reshape(vec, (var.NZ, var.NY, var.NX), 'C') #[z][y][x]

# Saves matrix in file
# \param filename (str): filename to write to
# \param matr (list 2D): matrix to save
def matrix2csv(filename, matr):
    open(filename, 'w')
    np.savetxt(filename, matr, delimiter=var.SEP, fmt='%.10e')


# Read and return content from binary file
# \param filename (str): filename to read from
# \return vec (list 1D) array from file
def read_bin(filename):
    with open(filename, 'r') as bin_file:
        vec = np.fromfile(bin_file, 'd') #'d' for double
    return vec


# Read and return content from csv file
# \param filename (str): filename to read from
# \return vec (list 1D) array from file
def read_csv(filename):
    with open(filename, 'r') as csv_file:
        vec = csv.reader(csv_file, delimeter=var.SEP)
    return vec


# Reads bin file and writes to csv
# \param filename_read (str): str of filename to read from
# \param filename_write (str): str of filename to write to
# \param normalize (bool): normalize array by U_MAX
# \param flip (bool): flip array vertically
def bin2csv(filename_read, filename_write, normalize = False, flip = True):
    vec = read_bin(filename_read)
    if(normalize):
        vec = normalize_array(vec)
    matr = array2matrix(vec)
    if(flip):
        matr = flip_matr_ver(matr)
    matrix2csv(filename_write, matr)

# Reads bin from files and writes to csv
# \param filenames_read (list(str)): list of filenames to read from
# \param filenames_write (list(str)): list of filenames to write to
# \param normalize (bool): normalize array by U_MAX
# \param flip (bool): flip array vertically
def bin2csv_files(filenames_read, filenames_write, normalize = False, flip = True):
    for i in range(0, len(filenames_read)):
        bin2csv(filenames_read[i], filenames_write[i], normalize, flip)


# Saves in csv file the values of the matrix in x=0.5 (normalized matrix)
# \param matr (list 2D): matrix of values
# \param filenames_write (str): filename to write to
# \param normalize (bool): normalize array by U_MAX
# \param flip (bool): flip array vertically
def save_hx(matr, filename_write, normalize = True, flip = False):
    vec = line_from_matrix(matr, 'x', var.NX//2)
    if(normalize):
        vec = normalize_array(vec)
    if(flip):
        vec = flip_matr_ver(vec)
    vec_norm = [(i/(var.NX-1), vec[i]) for i in range(0, len(vec))]
    matrix2csv(filename_write, vec_norm)


# Saves in csv file the values of the matrix in y=0.5 (normalized matrix)
# \param matr (list 2D): matrix of values
# \param filenames_write (str): filename to write to
# \param normalize (bool): normalize array by U_MAX
# \param flip (bool): flip array vertically
def save_hy(matr, filename_write, normalize = True, flip = False):
    vec = line_from_matrix(matr, 'y', var.NY//2)
    if(normalize):
        vec = normalize_array(vec)
    if(flip):
        vec = flip_matr_ver(vec)
    vec_norm = [(i/(var.NX-1), vec[i]) for i in range(0, len(vec))]
    matrix2csv(filename_write, vec_norm)


def save_vtk_3d(matrix, matrix_name, filename_write, points = True):
    """
        TODO: add support to POINT DATA
        Saves variables values to vtk format
        \param matrix (list(numpy array 3d): list of variable values 
        \param matrix_name (list(str)): list of names of variable in vtk file
        \param filename_write (str): filename to write to (NO EXTENSION)
        \param points (bool): if True, save as point centered data,
                              if False, save as cell centered data
    """
    dx, dy, dz = 1.0/var.NX, 1.0/var.NY, 1.0/var.NZ
    # grid
    x = np.arange(0, 1.0+0.1*dx, dx, dtype='float64')
    y = np.arange(0, 1.0+0.1*dy, dy, dtype='float64')
    z = np.arange(0, 1.0+0.1*dz, dz, dtype='float64')
    data = {}
    for i in range(0, len(matrix)):
        data[matrix_name[i]] = matrix[i]
    gridToVTK(filename_write, x, y, z, cellData=data)



if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)

    rho = read_bin(var.PATH_RHO_READ[-1])
    ux = read_bin(var.PATH_UX_READ[-1])
    uy = read_bin(var.PATH_UY_READ[-1])
    uz = read_bin(var.PATH_UZ_READ[-1])
    rho, ux, uy, uz = array2matrix3d(rho), array2matrix3d(ux), \
        array2matrix3d(uy), array2matrix3d(uz)
    rho, ux, uy, uz = np.swapaxes(rho, 0, 2), np.swapaxes(ux, 0, 2), \
        np.swapaxes(uy, 0, 2), np.swapaxes(uz, 0, 2) #[x][y][z] 
    ux, uy, uz = normalize_array(ux), normalize_array(uy), normalize_array(uz)
    pp.pprint(rho)
    pp.pprint(ux)
    '''
    # LID DRIVEN CAVITY
    #linear interpolation (for NZ even)
    tmp = (line_from_matrix3d(uz, ('x','z'), (var.NX//2,var.NZ//2)) \
        + line_from_matrix3d(uz, ('x','z'), (var.NX//2,var.NZ//2-1))) / 2
    curve_uz = [((0.5/var.NY + i/var.NY), tmp[i]) for i in range(0, len(tmp))]
    #linear interpolation (for NY even)
    tmp = (line_from_matrix3d(uy, ('x','y'), (var.NX//2,var.NY//2)) \
        + line_from_matrix3d(uy, ('x','y'), (var.NX//2,var.NY//2-1))) / 2
    curve_uy = [((0.5/var.NZ + i/var.NZ), tmp[i]) for i in range(0, len(tmp))]

    matrix2csv(var.PATH_UY_WRITE[0], curve_uy)
    matrix2csv(var.PATH_UZ_WRITE[0], curve_uz)
    '''

    '''
    # PARALLEL PLATES
    tmp = line_from_matrix3d(uz, ('x','z'), (var.NX//2,var.NZ//2))
    curve_uz = [(i/(var.NY-1), tmp[i]) for i in range(0, len(tmp))]
    matrix2csv(var.PATH_UZ_WRITE[0], curve_uz)
    '''
    
    '''
    # SAVE TO VTK
    for i in range(0, len(var.PATH_WRITE_VTK)):
        rho = read_bin(var.PATH_RHO_READ[i])
        ux = read_bin(var.PATH_UX_READ[i])
        uy = read_bin(var.PATH_UY_READ[i])
        uz = read_bin(var.PATH_UZ_READ[i])
        save_vtk_3d([rho, ux, uy, uz], ["density", "ux", "uy", "uz"], \
            var.PATH_WRITE_VTK[i], points=False)
        print("Terminou passo de tempo", i)
    '''

    '''
    bin2csv_files(var.PATH_RHO_READ, var.PATH_RHO_WRITE, normalize=False, flip=True)
    bin2csv_files(var.PATH_UX_READ, var.PATH_UX_WRITE, normalize=True, flip=True)
    bin2csv_files(var.PATH_UY_READ, var.PATH_UY_WRITE, normalize=True, flip=True)
    matr_ux = array2matrix(read_bin(var.PATH_UX_READ[0]))
    save_hx(matr_ux, var.PATH_DATA + var.ID_SIM + "_ux_c.csv", normalize=True)
    matr_uy = array2matrix(read_bin(var.PATH_UY_READ[0]))
    save_hy(matr_uy, var.PATH_DATA + var.ID_SIM + "_uy_c.csv", normalize=True)

    for i in range(0, len(var.PATH_UX_READ)):
        filename_read = var.PATH_UX_READ[i]
        filename_write = var.PATH_UX_WRITE[i]
        vec = read_bin(filename_read)
        matr = array2matrix(vec)
        save_hx(matr, filename_write, flip=False)
    
    for i in range(0, len(var.PATH_UY_READ)):
        filename_read = var.PATH_UY_READ[i]
        filename_write = var.PATH_UY_WRITE[i]
        vec = read_bin(filename_read)
        matr = array2matrix(vec)
        save_hx(matr, filename_write, flip=True)
    '''


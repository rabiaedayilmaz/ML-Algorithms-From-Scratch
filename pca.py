import numpy as np

"""
    For calculation of eigenvalue - eigenvector and determinant
    Thanks to: https://github.com/LucasBN/Eigenvalues-and-Eigenvectors/blob/master/main.py 
"""

dataset = [
    [5, 45, 12, 23],
    [4, 56, 42, 12],
    [4, 5, 18, 32],
    [19, 56, 23, 45],
]

class DatasetChecker:
    def __init__(self, data):
        self.data_type = type(data)
        self.data_type_checker(self.data_type)
        self.check_shape(data)

    def data_type_checker(self, data_type):
        """
        Raise error if not data type is not 'class <list>'

        :param data_type:
        """
        if data_type != list:
            raise Exception(f"Can not create dataset from {data_type} object. You have to pass list.")

    def check_shape(self, data):
        """
        Raise an error if not data is square matrix

        :param data:
        """
        row_length = len(data)
        col_length = len(data[0])
        if row_length != col_length:
            raise Exception("Length of all rows and columns must be the same! You have to pass square matrix.")

    def __str__(self):
        return f"\nChecks the data whether is not a square matrix or not a list"

class EigenValuesVectors:
    def __init__(self):
        """
        inherits the init properties of PCA class that calculates P values
        """
        self.dataset = dataset

    def get_cofactor(self, matrix, startrow, endcol):
        """
        Calculates cofactor of given matrix to find sub determinant
        Cofactor of matrix means that a sub matrix with excluded rows and columns

        For further info check out how to find a determinant: https://www.algebrapracticeproblems.com/cofactor-expansion/

        :param matrix: A list of lists of 'float'
        :param startrow: An 'int'
        :param endcol: An 'int'
        :return: A list of lists of 'float' for cofactor of give matrix
        """
        return [(row[:endcol] + row[endcol+1:]) for row in (matrix[:startrow] + matrix[startrow+1:])]

    def find_determinant(self, matrix, excluded=1):
        """
        Calculates determinant of a given matrix

        :param matrix: A list of lists of 'float'
        :param excluded: An 'int'
        :return: determinant for given matrix
        """
        s = 0
        dimensions = [len(matrix), len(matrix[0])]

        # base case - if it is 2x2 matrix
        if dimensions == [2, 2]:
            return excluded * (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        else:
            # recursive process through base case
            for currentcol in range(len(matrix)):
                subdet = self.find_determinant(self.get_cofactor(matrix, 0, currentcol))
                sign = (-1) ** currentcol
                s += sign * (matrix[0][currentcol] * subdet)
            return s

    def list_multiply(self, list1, list2):
        """
        Multiplies and them sums diagonal components while calculating determinant equation

        :param list1: A list of 'float'
        :param list2: A list of 'float'
        :return: A list of 'float' for multiplication and summation of lists
        """
        if type(list1) != list and type(list2) != list:
            list1 = [list1]
            list2 = [list2]
        result = [0 for _ in range(len(list1) + len(list2) - 1)]
        for i in range(len(list1)):
            for j in range(len(list2)):
                result[i + j] += list1[i] * list2[j]
        return result

    def list_add(self, list1, list2, sub=1):
        """
        Adds or subtracts result of diagonals due to 'sub' parameter for determinant equation

        :param list1: A list of 'float'
        :param list2: A list of 'float'
        :param sub: An 'int' to determine whether summation or subtraction
        :return: A list of 'float' for added lists for determinant calculation
        """
        return [i + j * sub for i, j in zip(list1, list2)]

    def determinant_equation(self, matrix, excluded=[1, 0]):
        """
        :param matrix: A list of lists of 'float'
        :param excluded: A list of 'int'
        :return: A list of 'float'
        """
        dimensions = [len(matrix), len(matrix[0])]
        if dimensions == [2, 2]:
            tmp = self.list_add(self.list_multiply(matrix[0][0], matrix[1][1]),
                                self.list_multiply(matrix[0][1], matrix[1][0]), sub=-1)
            return self.list_multiply(tmp, excluded)
        else:
            new_matrices = []
            excluded = []
            exclude_row = 0
            for exclude_column in range(dimensions[1]):
                tmp = []
                excluded.append(matrix[exclude_row][exclude_column])
                for row in range(1, dimensions[0]):
                    tmp_row = []
                    for col in range(dimensions[1]):
                        if (row != exclude_row) and (col != exclude_column):
                            tmp_row.append(matrix[row][col])
                    tmp.append(tmp_row)
                new_matrices.append(tmp)

            determinant_equations = [self.determinant_equation(new_matrices[j], excluded[j]) for j in range(len(new_matrices))]
            dt_equation = [sum(i) for i in zip(*determinant_equations)]
            return dt_equation

    def idenity_matrix(self, dimensions):
        """
        Return identity matrix for given dimensions

        :param dimensions: A list of 'int'
        :return: A list of lists of 'float' for identity matrix for given dimensions
        """
        id_matrix = [[0 for i in range(dimensions[1])] for j in range(dimensions[0])]
        for i in range(dimensions[0]):
            id_matrix[i][i] = 1
        return id_matrix

    def characteristic_equation(self, matrix):
        """
        Return characteristic equation for given matrix

        :param matrix: A list of lists of 'float'
        :return: A list of lists of lists of 'float'
        """
        dimensions = [len(matrix), len(matrix[0])]
        return [[[a, -b] for a, b in zip(i, j)] for i, j in zip(matrix, self.idenity_matrix(dimensions))]

    def find_eigenvalues(self, matrix):
        """
        Return eigenvalues of given matrix

        :param matrix: A list of lists of 'float'
        :return: A list of 'float' eigenvalues of given matrix
        """
        dt_eq = self.determinant_equation(self.characteristic_equation(matrix))
        return np.roots(dt_eq)
        #return np.linalg.eigvals(matrix)

    def __str__(self):
        return f"\nCalculates eigenvalues/vectors for the given matrix:\n{self.dataset}"


class PCA(DatasetChecker, EigenValuesVectors):
    def __init__(self, dataset):
        """
        Given dataset declares dimensions of dataset (rows, columns) as (n_samples, n_dim)
        and initializes a covariance matrix with zeros.

        :param dataset: A list of lists of 'float'
        """
        super().__init__(dataset)
        self.n_dim = len(dataset)
        self.n_samples = len(dataset[0])
        self.dataset = dataset
        self.cov = [[0] * self.n_dim] * self.n_dim
        self.pretty_print_pca_values()

    def compute_mean(self, samples):
        """
        Return mean values

        :param samples: A list of lists of 'float'
        :return: A 'int' value for mean
        """
        return sum(samples) / self.n_samples

    def compute_covariance_matrix(self, matrix):
        """
        Given dataset matrix, firstly computes mean value of each row,
        and then calculates covariance matrix.

        :param matrix: A list of lists of 'float'
        :return: A list of lists of 'float' for covariance matrix and a 'float' list for mean values
        """
        # compute averages
        avg = [self.compute_mean(sample) for sample in matrix]
        print("[INFO] Averages of samples: ", avg)

        # compute covariance matrix
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                var = 0
                for d in matrix:
                    var += (d[i] - avg[i]) * (d[j] - avg[j])
                var /= (self.n_samples - 1)
                self.cov[i][j] = var
        print("[INFO] Covariance matrix", self.cov)
        return self.cov, avg

    def normalize_vector(self, matrix):
        """
        Return normalized given matrix.

        :param matrix:  A list of lists of 'float'
        :return: normalized matrix vector
        """
        # initialize a zeros vector sam shape as matrix
        sum_vector = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]

        # take the summation of squares of each value and divide each component of matrix with root of summation
        for i in range(len(matrix)):
            vector = matrix[i]
            for j in range(len(vector)):
                val = vector[j]
                sum_vector[i][j] += val**2
            denominator = [sum(a)**(1/2) for a in sum_vector]
        normalized = [[matrix[i][j]/denominator[i] for i in range(len(matrix))] for j in range(len(matrix[0]))]
        print("[INFO] Normalized vector: ", normalized)
        return normalized

    def take_transpose(self, matrix):
        """
        Return transposed given matrix

        :param matrix:  A list of lists of 'float'
        :return:  A list of lists of 'float' for transposed matrix
        """
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = matrix[j][i]
        return matrix

    def matrix_multiplication(self, m1, m2):
        """
        Do matrix multiplication for given matrices

        :param m1: A list of lists of 'float' for first matrix (n x n)
        :param m2: A list of lists of 'float' for second matrix (n x n)
        :return: A list of lists of 'float' for matrix multiplication of two matrices
        """
        # create zeros matrix same shape as (n x n)
        result = [[0 for i in range(len(m2))] for j in range(len(m1))]
        # iterate through rows of first matrix
        for i in range(len(m1)):
            # iterate through cols of second matrix
            for j in range(len(m2[0])):
                # iterate through rows of second matrix
                for k in range(len(m2)):
                    result[i][j] += m1[i][k] * m2[k][j]
        return result

    def center_vals(self, matrix, avg):
        """
        :param matrix: A list of lists of 'float'
        :param avg: A list of 'float'
        :return: A list of lists of 'float' a matrix that has centered values

        Centers the values of matrix by subtraction mean value from each of them
        """
        for i in range(len(matrix)):
            vector = matrix[i]
            for j in range(len(vector)):
                value = vector[j]
                c = value - avg[j]
                matrix[i][j] = c
        return matrix

    def derive_pca_values(self, matrix, normalized_vectors, avg):
        """
        :param matrix: A list of lists of 'float'
        :param normalized_vectors: A list of lists of 'float'
        :param avg: A list of 'float'
        :return: A list of lists of 'float' for P value that represents importance of features

        Calculates P value by computing (normalized_vectors).T * [x1 - x_avg, y1 - y_avg ...]
        """
        transposed_matrix = self.take_transpose(normalized_vectors)
        centered_vector = self.center_vals(matrix, avg)
        p_val = self.matrix_multiplication(transposed_matrix, centered_vector)
        return p_val

    def pca_results(self, matrix):
        """
        :param matrix: A list of lists of 'float'
        :return: A list of lists of 'float' for all P values

        Calculates P values by using mean values and normalized vector of covariance matrix
        """
        cov_matrix, avg = self.compute_covariance_matrix(matrix)
        normalized_vector = self.normalize_vector(cov_matrix)
        pca_vals = self.derive_pca_values(matrix, normalized_vector, avg)
        return pca_vals

    def pretty_print_pca_values(self):
        """
        :param pca_vals: A list of lists of 'float'
        :return: None

        Pretty print for P values in matrix notation
        [P11 P12 P13]
        [P21 P22 P23]
        [P31 P32 P33] ...
        """
        pca_results = self.pca_results(self.dataset)
        line = "*.+'" * 42
        print("[INFO] Eigenvalues: ", self.find_eigenvalues(self.dataset))
        print(line)
        print("[INFO] PCA values: ", pca_results)
        print(line)

        for i in range(len(pca_results)):
            vec = pca_results[i]
            for j in range(len(vec)):
                val = vec[j]
                print("P" + str(i) + str(j) + ": " + str(val))

    def __str__(self):
        return f"\nCalculates PCA values for the given matrix:\n{self.dataset}"

print(PCA(dataset))

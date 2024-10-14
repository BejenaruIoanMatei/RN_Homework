
## ----1-----
# Parsing the System of Equations (1 point)
#voi folosi file.txt unde voi avea scris sistemul de ecuatii

def parse_equations(file_path):
    A = [] # matricea
    B = [] # vectorul
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip() ## pentru a elimina spatiile
            lhs, rhs = line.split('=') # avem partea st a ecuatiei si partea dr
            coeffs = [0, 0, 0]
            lhs = lhs.replace('-', '+-')
            
            #pentru partea stanga
            terms = lhs.split('+')
            
            for term in terms:
                term = term.strip() 
                if term == '':
                    continue
                if 'x' in term:
                    coeff = term.replace('x', '').strip()
                    coeffs[0] = int(coeff) if coeff not in ['', '+', '-'] else int(coeff + '1')
                elif 'y' in term:
                    coeff = term.replace('y', '').strip() 
                    coeffs[1] = int(coeff) if coeff not in ['', '+', '-'] else int(coeff + '1')
                elif 'z' in term:
                    coeff = term.replace('z', '').strip() 
                    coeffs[2] = int(coeff) if coeff not in ['', '+', '-'] else int(coeff + '1')
            A.append(coeffs)
            B.append(int(rhs.strip()))
    
    return A, B

#ca sa comentez blocul
""" 
if __name__ == "__main__":
    file_path = 'file.txt'  # fisierul cu ecuatiile
    A, B = parse_equations(file_path)
    
    print("matricea: ")
    for row in A:
        print(row)
    print("\n")
    print("vectorul: ")
    print(B)
"""

###------2------ Matrix and Vector Operations

import math

# Determinantul unei mat 2x2 sau 3x3
## modificata ca sa o pot folosi si la det(minor) care e 2x2
def determinant(A):
    if len(A) == 2:  # pt 2x2
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    elif len(A) == 3:  # pt 3x3
        return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    else:
        raise ValueError("Matricea nu este nici 2x2, nici 3x3")

# sum pe diagonala principala
def trace(A):
    return A[0][0] + A[1][1] + A[2][2]

# Norma eucl a vectorului B
def vector_norm(B):
    return math.sqrt(B[0]**2 + B[1]**2 + B[2]**2)

# Transpunerea unei matrice 3x3
def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]

# O matrice 3x3 inmultita cu un vector 
def matrix_vector_multiplication(A, B):
    return [
        A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2],
        A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2],
        A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2]
    ]

"""if __name__ == "__main__":
    # o sa iau matricea si vectorul din punctul anterior
    file_path = 'file.txt'  # fisierul cu ecuatiile
    A, B = parse_equations(file_path)
    
    print("Det lui A:", determinant(A))
    print("Trace ul lui A:", trace(A))
    print("Norma vectorului B:", vector_norm(B))
    
    A_transpose = transpose(A)
    print("Transpose de A:")
    for row in A_transpose:
        print(row)
    
    result_vector = matrix_vector_multiplication(A, B)
    print("Rez inmultirii lui A cu B:", result_vector)
    """

### ------3------- Solving using Cramer ---

# inlocuiesc o coloana din matrice cu un vector
def replace_column(A, B, col_index):
    A_copy = [row[:] for row in A]
    
    for i in range(3):
        A_copy[i][col_index] = B[i]
    
    return A_copy

# functia in sine pentru Cramer
def cramer(A, B):
    det_A = determinant(A)
    if det_A == 0:
        raise ValueError("nu are sol unica daca avem determinantul = 0")
    # inloc prima, a doua , a treia col pe rand pentru A_x, A_y, A_z
    A_x = replace_column(A, B, 0)
    det_Ax = determinant(A_x)
    
    A_y = replace_column(A, B, 1)
    det_Ay = determinant(A_y)
    
    A_z = replace_column(A, B, 2)
    det_Az = determinant(A_z)
    
    x = det_Ax / det_A # ca numar real
    y = det_Ay / det_A
    z = det_Az / det_A
    ## astea vor fi solutiile
    
    return x, y, z


"""if __name__ == "__main__":
    file_path = 'file.txt'
    A, B = parse_equations(file_path)
    print("matricea: ")
    for row in A:
        print(row)
    print("\n")
    print("vectorul: ")
    print(B)
    x, y, z = cramer(A, B)
    print(f"Soluția sistemului: x = {x}, y = {y}, z = {z}")
    """

# ------4------- Solving using Inversion

# fct pentru minor
def minor_matrix(A, i, j):
    return [row[:j] + row[j+1:] for row in (A[:i] + A[i+1:])]

# fct pt cofactor
def cofactor(A, i, j):
    minor = minor_matrix(A, i, j)
    return ((-1) ** (i + j)) * determinant(minor)

# cofactor matrix (−1)i+j × det(Mij )

def cofactor_matrix(A):
    cofactor_mat = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(cofactor(A, i, j))
        cofactor_mat.append(row)
    return cofactor_mat

#inversam matricea
def inverse_matrix(A):
    det_A = determinant(A)
    
    if det_A == 0:
        raise ValueError("det lui A este 0")
    
    cofactor_mat = cofactor_matrix(A)
    
    # transpusa cofactorilor = adjugata
    adjugate = transpose(cofactor_mat)
    
    # inversa este adjugata / det(A) , cu det A neaparat != 0
    inverse = [[adjugate[i][j] / det_A for j in range(3)] for i in range(3)]
    
    return inverse

def matrix_multiplication(A, B):
    return [
        A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2],
        A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2],
        A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2]
    ]

if __name__ == "__main__":
    file_path = 'file.txt'
    A, B = parse_equations(file_path)
    print("matricea: ")
    for row in A:
        print(row)
    print("\n")
    print("vectorul: ")
    print(B)
    
    x, y, z = cramer(A, B)
    print("CRAMER")
    print(f"Soluția sistemului: x = {x}, y = {y}, z = {z}")
    print("\n")
    
    print("INVERSA")
    
    A_inv = inverse_matrix(A)
    print("Inversa lui A:")
    for row in A_inv:
        print(row)
        
    result_vector = matrix_multiplication(A_inv, B)
    print(f"Soluția sistemului folosind inversa: x = {result_vector[0]}, y = {result_vector[1]}, z = {result_vector[2]}")

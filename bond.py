import numpy as np

query_attr = np.array([[1, 0, 1, 0], 
                       [0, 1, 1, 0], 
                       [0, 1, 0, 1], 
                       [0, 0, 1, 1]])

query_access = np.array([[15, 20, 10], 
                         [5, 0, 0], 
                         [25, 25, 25], 
                         [3, 0, 0]])

dim = len(query_attr[0])
aa_matrix = np.zeros((dim,dim))

def position_checker(positions, i, k):
  aa_elem = []
  for j in positions[0]:
    aa_elem.append(np.sum(query_access[j, :]))
  aa_matrix[i,k] = sum(aa_elem)

def dot_prod(vec_a, vec_b):
  return np.dot(vec_a.flatten(), vec_b.flatten())

## AA_MARIX GENERATION
for i in range(dim):
  for k in range(dim):
    if i == k:
      attr = query_attr[:, k].flatten()
      positions = np.where(attr == 1)
      position_checker(positions, i, k)
    else:
      attr_i = query_attr[:, i].flatten()
      attr_k = query_attr[:, k].flatten()
      positions = np.where((attr_i == 1) & (attr_k == 1))
      position_checker(positions, i, k)

print("AA matrix")
print(aa_matrix)

# BOND ENERGY ALGORITHM - GET INITIAL ca_matrix
ca_matrix = aa_matrix[:, :2]
ca_matrix = np.append(np.zeros((dim,1)), ca_matrix, axis=1)
ca_matrix = np.append(ca_matrix,np.zeros((dim,1)), axis=1)

# BOND ENERGY ALGORITHM - ca_matrix GENERATION
for k in range(2, len(aa_matrix)):
  i, j = 0, 1
  get_values = []
  while j < ca_matrix.shape[1]:
    #print(i,k,j, len(ca_matrix))
    get_values.append(2 * dot_prod(ca_matrix[:, i], aa_matrix[:, k]) + 
                      2 * dot_prod(aa_matrix[:, k], ca_matrix[:, j]) - 
                      2 * dot_prod(ca_matrix[:, i], ca_matrix[:, j]))  
    i = j
    j+=1
  pos = np.argmax(get_values) + 1
  ca_matrix = np.insert(ca_matrix, pos, aa_matrix[k], axis=1)

# BOND ENERGY ALGORITHM - REMOVES ZEROS COLUMNS
ca_matrix = np.delete(ca_matrix, 0, axis=1)
ca_matrix = np.delete(ca_matrix, ca_matrix.shape[1]-1, axis=1)

print("\nCA matrix")
print(ca_matrix)

column_indices = []
outArr = []

for i in range(0, dim):
  for j in range(0, dim):
    if j not in column_indices and np.array_equal(ca_matrix[:, i], aa_matrix[:, j]):
      column_indices.append(j)

for i in column_indices:
  outArr.append(ca_matrix[i, :])

ca_matrix_adjusted = np.array(outArr)

print("\nCA matrix (adjusted)")
print(ca_matrix_adjusted)  
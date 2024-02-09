import numpy as np
import copy

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

# Adjust rows. Since we didn't keep track of them, this requires some searching.
column_indices = []
outArr = []

for i in range(0, dim):
  for j in range(0, dim):
    if j not in column_indices and np.array_equal(ca_matrix[:, i], aa_matrix[:, j]):
      column_indices.append(j)

column_indices = np.array(column_indices)

for i in column_indices:
  outArr.append(ca_matrix[i, :])

ca_matrix_adjusted = np.array(outArr)

print("\nCA matrix (after row adjustment)")
print(ca_matrix_adjusted)  

# Paritioning Algorithm

# Helper that examines query and determines if it uses TQ and BQ attributes.
def check_use(query):

  accessTQ = False
  for i in att_TQ:
   if query[i] == 1:
    accessTQ = True
    break

  accessBQ = False
  for i in att_BQ:
   if query[i] == 1:
    accessBQ = True
    break
    
  return accessTQ, accessBQ

# Checks all split points and determines the best one.

# keep track of best z-score and the corresponding shift counter value and location on the diagonal.
best_i = 0
best_z = float('-inf')
best_shift = 0

# Count shifts and retain original attribute position to know when we've exhausted shifts.
column_indices_orig = copy.deepcopy(column_indices)
shift_counter = 0

# Run and keep shifting until we return to our start point.
print("\nStarting partition algorithm.")
while True:
 for i in range(0, dim - 1):

   # get attributes determined by split point along the diagonal.
   att_TQ = column_indices[:i+1]
   att_BQ = column_indices[min(i+1, dim-1):]

   # Reference: Page 108 of the textbook mentioned in the readme.
   query_TQ = [i for i in range(0, dim) if check_use(query_attr[i]) == (True, False)]
   query_BQ = [i for i in range(0, dim) if check_use(query_attr[i]) == (False, True)]
   query_OQ = [i for i in range(0, dim) if check_use(query_attr[i]) == (True, True)]

   val_CTQ = np.sum(query_access[query_TQ].flatten())
   val_CBQ = np.sum(query_access[query_BQ].flatten())
   val_COQ = np.sum(query_access[query_OQ].flatten())

   z_value = val_CTQ * val_CBQ - val_COQ ** 2

   print(f"\nPartition after {i+1},{i+1}: z-value is {z_value}")

   if z_value > best_z:
     best_z = z_value
     best_i = i
     best_shift = shift_counter
   
 # Perform the shift. This should correspond to moving the rows/columns around as the textbook describes. 
 column_indices = np.roll(column_indices, -1)
   
 # Exit the loop if we've exhausted our shifts.
 if np.array_equal(column_indices, column_indices_orig):
  print("\nStopping algorithm because we've exhausted our shifts.")
  break
   
 shift_counter += 1
 print(f"\nPerformed shift. Number of shifts performed so far: {shift_counter}")

print(f"\nThe optimal split requires {best_shift} shifts and occurs along the diagonal after {best_i + 1},{best_i + 1}.")

# We show the split. Featuring hacky print formatting! Normally I'd shove this all in a dataframe and print that but I don't want to force another dependency on you guys.
print("\nThe optimal partitioning looks like this:\n")

# We adjust the CA matrix by the best number of shifts.
for i in range(0, best_shift):
  ca_matrix_adjusted = np.roll(ca_matrix_adjusted, -1, axis = 1)
  ca_matrix_adjusted = np.roll(ca_matrix_adjusted, -1, axis = 0)

# convert array to string to be able to add breaks
splitArr = ca_matrix_adjusted.astype(int)
splitArr = splitArr.astype(str)

# Add required spaces to make it look nice.
column_spacing = []
for i in range(0, dim):
    column_spacing.append(np.max(np.array([len(x) for x in splitArr[:, i]])))

for i in range(0, dim):
  for j in range(0, dim):
    splitArr[i, j] += " " * (column_spacing[i] - len(splitArr[i, j]))

# insert breaks to show split
splitArr = np.insert(splitArr, best_i + 1, "|", axis = 1)
splitArr = np.insert(splitArr, best_i + 1, "-", axis = 0)

# Add spaces to breaks
for j in range(0, dim):
  splitArr[best_i + 1, j] += " " * (column_spacing[best_i + 1] - len(splitArr[best_i + 1, j]))

# print without brackets or apostrophes.
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in splitArr]))
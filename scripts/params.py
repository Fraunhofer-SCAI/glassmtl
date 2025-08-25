"""
Parameter file
"""

# Random seed
SEED = 42  #1 #100


# Elements for oxide glasses available in the glassmodel 
# (ordered with respect to the atomic numbers)
# Padding element 'X' must be at position 0
ELEMENTS = [  # 67 elements (including X)
    'X', 'Li', 'Be', 'B', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 
    'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 
    'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
    'Re', 'Hg', 'Tl', 'Pb', 'Bi', 'Th', 'U']


# Maximum number of processes to be run in parallel
# Set lower value if you run out of memory
MAX_N_PROCESSES = 80

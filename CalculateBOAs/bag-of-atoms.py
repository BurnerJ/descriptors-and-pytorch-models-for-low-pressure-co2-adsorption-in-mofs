"""

@Author: Jake Burner, University of Ottawa, May, 2020

The purpose of the following code is to generate bags for the bag-of-atoms descriptor, as described in the corresponding publication. Specifically, this code takes a .cif file and separates the atoms contained in it into 6 x 6 x 6 cuboids (or "bags"). This code does not completely calculate the descriptor, the purpose of this code is to ensure the bags are generated in a way that the developed models will recognize.

For instructions on using this code, please read the corresponding README.

"""

import numpy as np
import pandas as pd
import os

# The directory of the CIF files
directory_in_str = 'C:/Users/Jake/OneDrive - University of Ottawa/Desktop/QSPR Codes/cifs'
directory = os.fsencode(directory_in_str)

# Name of CSV file to store descriptors
csv = 'atom-bins.csv'

# For every file in the directory...
for file in os.listdir(directory):

    # Generate the "bag" as a nested list
    bag_of_atoms = np.zeros((6, 6, 6), str)
    bag_of_atoms = bag_of_atoms.tolist()

    # Set up a dataframe to store all 6*6*6 bins of atoms, and generate column names depending on which section of the unit cell each atom is located in
    bag_of_atoms_df = pd.DataFrame(index=np.arange(1), columns=np.arange(216))
    column_names = []
    for i in range(6):
        for n in range(6):
            for m in range(6):
                column_names.append('bin {}{}{}'.format(i,n,m))
    bag_of_atoms_df.columns = column_names

    # Read the CIF line-by-line
    filename = os.fsdecode(file)
    keyword_start = '_atom_type_partial_charge'
    keyword_end = 'loop_'
    cif_file = open('{}/{}'.format(directory_in_str, filename), 'r')
    line = cif_file.readline().split('data_')
    MOF_name = line[1]

    while keyword_start not in line:
        line = cif_file.readline()

    counter = 0

    while keyword_end not in line:
        # Except an index error since there is a blank line before end keyword appears
        try:
            # Split the line into a list
            line = cif_file.readline().split()
            atom_type = line[1]
            x = float(line[3])
            y = float(line[4])
            z = float(line[5])
            
            posx = None
            posy = None
            posz = None
            
            # A counter to count the number of total framework atoms
            counter += 1

            # "Cut" the unit cell into sixths in all dimensions and assign an x,y,z position (0, 1, 2, 3, 4, or 5) for each atom
            i = 0
            while (posx == None or posy == None or posz == None):
                if i*(1/6) <= x < (i+1)*(1/6):
                    posx = i
                if i*(1/6) <= y < (i+1)*(1/6):
                    posy = i
                if i*(1/6) <= z <= (i+1)*(1/6):
                    posz = i
                i += 1

            # Add it to the bag and corresponding "bin"
            bag_of_atoms[posx][posy][posz] += (' ' + atom_type)
            bag_of_atoms_df['bin {}{}{}'.format(posx, posy, posz)] = bag_of_atoms[posx][posy][posz]

        except IndexError:
            pass

    # Put the data into a csv file and make the index the name of the CIF file
    bag_of_atoms_df.index = [MOF_name]
    # Add the number of atoms to the dataframe
    bag_of_atoms_df['MOF name'] = MOF_name
    bag_of_atoms_df['num_atoms'] = counter
    
    if os.path.exists('{}/{}'.format(directory_in_str, csv)):
        data = pd.read_csv('{}/{}'.format(directory_in_str, csv), index_col=0)
        bag_of_atoms_df = pd.concat([data, bag_of_atoms_df])
        bag_of_atoms_df.to_csv('{}/{}'.format(directory_in_str, csv))
    else:
        bag_of_atoms_df.to_csv('{}/{}'.format(directory_in_str, csv))

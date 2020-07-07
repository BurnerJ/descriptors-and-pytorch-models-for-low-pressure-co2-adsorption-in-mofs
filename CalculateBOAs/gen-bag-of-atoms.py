"""

@author: Jake Burner, University of Ottawa, May, 2020

The purpose of the following code is to generate the bag-of-atoms descriptor from the atom bags generated from the 'bag-of-atoms.py' code.

For instructions on using this code, please read the corresponding README.

"""

import numpy as np
import pandas as pd
import os

# The directory of the CIF files
directory_in_str = 'C:/Users/Jake/OneDrive - University of Ottawa/Desktop/QSPR Codes/cifs'
directory = os.fsencode(directory_in_str)

epsilon_dict = {
    'O': 0.06,
    'C': 0.105,
    'Zn': 0.124,
    'N': 0.069,
    'H': 0.044,
    'Fe': 0.013,
    'Cl': 0.227,
    'Cu': 0.005,
    'S': 0.274,
    'Co': 0.014,
    'F': 0.05,
    'Ni': 0.015,
    'In': 0.599,
    'I': 0.339,
    'V': 0.016,
    'Cd': 0.228,
    'Br': 0.251,
    'Cr': 0.015,
    'Mn': 0.013,
    'Zr': 0.069,
    'P': 0.305,
    'Ba': 0.364,
    'Mg': 0.111,
    'Al': 0.505,
}

sigma_dict = {
    'O': 3.1181,
    'C': 3.4309,
    'Zn': 2.4616,
    'N': 3.2607,
    'H': 2.5711,
    'Fe': 2.5943,
    'Cl': 3.5164,
    'Cu': 3.1137,
    'S': 3.5948,
    'Co': 2.5587,
    'F': 2.997,
    'Ni': 2.5248,
    'In': 3.9761,
    'I': 4.009,
    'V': 2.801,
    'Cd': 2.5373,
    'Br': 3.732,
    'Cr': 2.6932,
    'Mn': 2.638,
    'Zr': 2.7832,
    'P': 3.6946,
    'Ba': 3.299,
    'Mg': 2.6914,
    'Al': 4.0082,
}

# Name of CSV file to store descriptors and one to read the atom bin data
csv1 = 'atom-bins.csv'
csv2 = 'descriptors.csv'

# Read CSV file
bag_of_atoms_df = pd.read_csv('{}/{}'.format(directory_in_str, csv1), index_col=0)

for MOF in range(bag_of_atoms_df.shape[0]):
    # Get the MOF name and make it the index name
    MOF_name = bag_of_atoms_df.iloc[int(MOF), bag_of_atoms_df.columns.get_loc('MOF name')]

    # Generate the "bag" as a nested list
    bag_of_atoms = np.zeros((6, 6, 6), str)
    bag_of_atoms = bag_of_atoms.tolist()

    # Also generate the epsilon and sigma descriptor nested lists
    bag_of_epsilons = np.zeros((6, 6, 6), str)
    bag_of_sigmas = np.zeros((6, 6, 6), str)
    bag_of_epsilons = bag_of_epsilons.tolist()
    bag_of_sigmas = bag_of_sigmas.tolist()

    # Generate dataframe for descriptors
    descriptors_df = pd.DataFrame(index=np.arange(1))


    # Populate the "bag" from the CSV file
    for i in range(6):
        for n in range(6):
            for m in range(6):
                epsilon, sigma, epsilon_total, sigma_total = 0, 0, 0, 0
                # Read what's in bin [i][n][m]
                bag_of_atoms[i][n][m] = bag_of_atoms_df.iloc[int(MOF), bag_of_atoms_df.columns.get_loc('bin {}{}{}'.format(i,n,m))]
                # Obtain a list of atoms in bin [i][n][m]
                atoms_list = str(bag_of_atoms[i][n][m]).split()
                # Retrieve total number of atoms in each MOF
                num_atoms = bag_of_atoms_df.iloc[int(MOF), bag_of_atoms_df.columns.get_loc('num_atoms')]
                # If it's empty, epsilon, sigma = 0
                if 'nan' in atoms_list:
                    bag_of_epsilons[i][n][m] = 0
                    bag_of_sigmas[i][n][m] = 0
                # If it has atoms, compute the total epsilon and sigma
                else:
                    for atom in atoms_list:
                        epsilon = epsilon_dict[atom]
                        sigma = sigma_dict[atom]
                        epsilon_total += epsilon
                        sigma_total += sigma
                    # Then, assign each value to the specific bin. The descriptor is the sum of each parameter normalized by the number of framework atoms in the MOF
                    bag_of_epsilons[i][n][m] = float(epsilon_total)/num_atoms
                    bag_of_sigmas[i][n][m] = float(sigma_total)/num_atoms
                # Add the descriptors to the descriptors dataframe
                descriptors_df['epsilon bin {}{}{}'.format(i,n,m)] = ('{:8.8f}'.format(bag_of_epsilons[i][n][m]))
                descriptors_df['sigma bin {}{}{}'.format(i,n,m)] = ('{:8.8f}'.format(bag_of_sigmas[i][n][m]))

    # make the index the name of the CIF file
    descriptors_df.index = [MOF_name]

    # Write the data to a csv file
    if os.path.exists('{}/{}'.format(directory_in_str, csv2)):
        descriptors_df.to_csv('{}/{}'.format(directory_in_str, csv2), mode='a', header=False)
    else:
        descriptors_df.to_csv('{}/{}'.format(directory_in_str, csv2))


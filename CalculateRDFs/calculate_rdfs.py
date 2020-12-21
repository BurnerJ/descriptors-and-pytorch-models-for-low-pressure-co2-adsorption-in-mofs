'''
July 2, 2020

@Author: Jun Luo

Modified by Jake Burner, July 3, 2020

'''
import numpy as np
import multiprocessing as mp
from glob import glob
from CifFile import ReadCif
from atomic_property_dict import apd
from itertools import product, combinations, combinations_with_replacement
from datetime import datetime

########################### USER MUST DEFINE THESE ###########################

# Where your cifs are located (src) and desired path for csv
src = "OneDrive/Documents/RDFs/cifs"
dst = "OneDrive/Documents/RDFs/cifs/RDFs.csv"

# Number of cores for calculations
n_cores = 3

# Smooth parameter (B) and factor (f)
smooth = -10
factor = 0.001

# Desired distance bins (in this case with linear increase in bin size from 2 to 30 A)
bins = np.arange(113, dtype=np.float64)
bins[0] = 2.0
step = 0.004425

for i in range(1, 113):
    bins[i] = bins[i-1] + step
    step += 0.004425

n_bins = len(bins)

# Properties desired for the RDFs
prop_names = ["electronegativity", "hardness", "vdWaalsVolume"]
prop_list = [apd[name] for name in prop_names]
n_props = len(prop_names)

###############################################################################

super_cell = np.array(list(product([-1, 0, 1], repeat=3)), dtype=float)

csv_header = [f"RDF_{prop}_{r:.2f}" for prop in prop_names for r in bins]
csv_header.insert(0, "Structure_Name")


def main(name):
    mof = ReadCif(name)
    mof = mof[mof.visible_keys[0]]

    elements = mof["_atom_site_type_symbol"]
    n_atoms = len(elements)

    prop_dict = {}
    for a1, a2 in combinations_with_replacement(set(elements), 2):
        prop_arr = [prop[a1] * prop[a2] for prop in prop_list]
        prop_dict[(a1, a2)] = prop_arr
        if a1 != a2:
            prop_dict[(a2, a1)] = prop_arr

    la = float(mof["_cell_length_a"])
    lb = float(mof["_cell_length_b"])
    lc = float(mof["_cell_length_c"])
    aa = np.deg2rad(float(mof["_cell_angle_alpha"]))
    ab = np.deg2rad(float(mof["_cell_angle_beta"]))
    ag = np.deg2rad(float(mof["_cell_angle_gamma"]))
    # If volume is missing from .cif, calculate it.
    try:
        cv = float(mof["_cell_volume"])
    except KeyError:
        cv = la * lb * lc * math.sqrt(1 - (math.cos(aa)) ** 2 -
                (math.cos(ab)) ** 2 - (math.cos(ag)) ** 2 +
                (2 * math.cos(aa) * math.cos(ab) * math.cos(ag)))


    frac2cart = np.zeros([3, 3], dtype=float)
    frac2cart[0, 0] = la
    frac2cart[0, 1] = lb * np.cos(ag)
    frac2cart[0, 2] = lc * np.cos(ab)
    frac2cart[1, 1] = lb * np.sin(ag)
    frac2cart[1, 2] = lc * (np.cos(aa) - np.cos(ab)*np.cos(ag)) / np.sin(ag)
    frac2cart[2, 2] = cv / (la * lb * np.sin(ag))

    frac = np.array([
        mof["_atom_site_fract_x"],
        mof["_atom_site_fract_y"],
        mof["_atom_site_fract_z"],
    ], dtype=float).T

    apw_rdf = np.zeros([n_props, n_bins], dtype=np.float64)
    for i, j in combinations(range(n_atoms), 2):
        cart_i = frac2cart @ frac[i]
        cart_j = (frac2cart @ (super_cell + frac[j]).T).T
        dist_ij = min(np.linalg.norm(cart_j - cart_i, axis=1))
        rdf = np.exp(smooth * (bins - dist_ij) ** 2)
        rdf = rdf.repeat(n_props).reshape(n_bins, n_props)
        apw_rdf += (rdf * prop_dict[(elements[i], elements[j])]).T
    apw_rdf = np.round(apw_rdf.flatten() * factor / n_atoms, decimals=12)

    return ("{}," * len(apw_rdf) + "{}\n").format(
        name.split('/')[-1], *apw_rdf.tolist())


if __name__ == "__main__":

    start = datetime.now()
    print(start)
    print("==================================================================================================================================")
    print("Start: ", start.strftime("%c"))
    print("")
    print("")
    print("Starting RDF calculations on structures in {}, using {} cores...".format(src, n_cores))
    print("RDFs will be written continuously to: {}".format(dst))

    with open(dst, 'w') as csv, mp.Pool(n_cores) as pool:
        csv.write(','.join(csv_header) + '\n')
        csv.flush()
        for results in pool.imap_unordered(main, glob(f"{src}/*.cif")):
            csv.write(results)
            csv.flush()

    print("")
    print("")
    print("Finished! RDFs have been saved to: {}".format(dst))
    end = datetime.now()
    print("End: ", end.strftime("%c"))
    print("==================================================================================================================================")

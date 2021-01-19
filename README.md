# descriptors-and-pytorch-models-for-low-pressure-co2-adsorption-in-mofs

The Python scripts included were written or modified in part by current/former members of the Woo Lab at the University of Ottawa: Jake Burner and Jun Luo. Specific contributions are included in each script.

The data used in our study (https://doi.org/10.1021/acs.jpcc.0c06334) (i.e. an example of how the descriptors should be named/ordered) is available at this link: https://1drv.ms/u/s!AtuVqcWZi8aAy11S2wxataTe8IMH. Within this zip folder there are three zip folders and one csv:
        -all_data.csv (the descriptor values AND target values (CO2 working capacity, CO2/N2 selectivity), as described in the paper)
        -dev_cifs.zip (the cifs we used in our development/validation set)
        -test_cifs.zip (the cifs we used in our test set)
        -train_cifs.zip (the cifs we used in our training set)

Included are Python3 scripts to do the following (any missing packages can be easily pip installed):

1. Calculate AP-RDF descriptors on a directory of cifs
2. Calculate the bag-of-atoms descriptor on a directory of cifs
3. Predict CO2 working capacity and CO2/N2 selectivity on a database of materials for which AP-RDF, chemical motifs, geometric, or bag-of-atoms descriptors have been calculated using models developed in https://doi.org/10.1021/acs.jpcc.0c06334.

IMPORTANT NOTES
=====================================================================================================================================================================

1. AP-RDF DESCRIPTOR CALCULATION

To use this code, go to the "CalculateRDFs" directory, and run the "calculate_rdfs.py" code. This code requires user modifications from lines 17-46. Instructions are commented in the code, but source (location of cifs) and destination (location and name of csv file) are required in addition to desired number of cores to use for the calculation, the smoothing (B) parameter value, and factor (f) value. The distance bins can be modified in this portion of the code as well. Finally, the desired properties for the RDFs must be specified here as well. The properties can be found in the atomic_property_dict.py file. By default, the code normalizes the RDFs by the total number of atoms in the structure.


=====================================================================================================================================================================

2. BAG-OF-ATOMS DESCRIPTOR CALCULATION

To calculate this descriptor, navigate to the CalculateBOAs directory and edit the "bag-of-atoms.py" code on line 16. The variable "directory_in_str" should be changed to the path to the cif files. Once this is done, run the code and it will generate a csv file ("atom-bins.csv") containing the 216 epsilon and 216 sigma "bags" with their corresponding atoms. Then, edit the "gen-bag-of-atoms.py" code on line 16. The variable "directory_in_str" should be changed to the path of the csv file created in the previous step (by default, the same directory as that containing the cifs). This will generate a new csv with the bag-of-atoms descriptor called "descriptors.csv" in the directory containing the cifs.

=====================================================================================================================================================================

3. USING THE PYTORCH MODELS TO PREDICT ADSORPTION PROPERTIES

Descriptors must be computed the same was as described in the publication for these models to be of any use and for this code to work. If the dimensions of the descriptors differ from what was done in this work, not only will the results be unreliable, but this code will not work. For this reason, it is suggested that for the AP-RDF descriptors (339 descriptor values per MOF) and bag-of-atoms descriptors (432 descriptor values per MOF), the included code be used as described in parts 1 and 2 of this README.NOTE: THE CSV CONTAINING THE USER'S DESCRIPTORS MUST MATCH THE DESCRIPTORS IN THE PROVIDED CSV FILE EXAMPLE (INCLUDING THE NAMING OF THE DESCRITPTORS). This also means that the descriptors (when combinations of descriptors are used) must be put in the following order in the user's csv file (the motifs descriptors in the order of the provided csv file example, the bag-of-atoms as generated above, the RDFs as generated above, and then the six geometric descriptors). All descriptors must be named the same way as they are named in the provided csv file example for the "load_pytorch" code to work without modification. One must first edit the load_pytorch.py code as follows (with acceptable input for these values is given as comments in the code):

The code requires three things to be specified from the user, starting on line 136 of the code:
a) the feature (descriptor) set
b) the target value
c) the name of the csv file containing the descriptor values, PLACED IN THE SAME DIRECTORY AS THE "load_pytorch.py" FILE!

The following is an example of output given by the program:

Start:  Wed May 20 15:40:00 2020
Device:  cpu

        Reading in data... This may take a few minutes depending on your device and the size of your CSV file.

        Feature/Descriptor set:  Bag of Atoms + APW-RDF 771 features

        Loading in PyTorch model...

        Making predictions on the dataset...

        Preparing the CSV file with results...

Successful termination.
End: Wed May 20 15:41:54 2020
Total time: 114.7 s

=====================================================================================================================================================================

Any questions on using the code included here may be directed to Jake Burner at jburn072@uottawa.ca.


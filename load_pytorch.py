"""

@author: Jake Burner, University of Ottawa, May, 2020

The purpose of the following code is to utilize PyTorch models developed by Burner et al. for predicting low-pressure CO2 adsorption properties (particularly CO2 working capacity and CO2/N2 selectivity) of MOFs.

For instructions on using this code, please read the corresponding README.

"""

import torch
import warnings
import sys
import os.path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Class for 3-layer models
class Net3(nn.Module):
    
    def __init__(self):
        super(Net3, self).__init__()

    #forward function applies activation function
    #to input and sends it to output (x is input tensor)
    def forward(self, x):
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.dropout(F.relu(self.hidden2(x)))
        x = self.dropout(F.relu(self.hidden3(x)))
        x = F.relu(self.output(x))
        return x
        
# Class for 2-layer models
class Net2(nn.Module):
    
    def __init__(self):
        super(Net2, self).__init__()
        
    #forward function applies activation function
    #to input and sends it to output (x is input tensor)
    def forward(self, x):
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.dropout(F.relu(self.hidden2(x)))
        x = F.relu(self.output(x))
        return x

def get_features(feature_set, data):

    geom_features = ["CO2_Surf_m2/g", "CO2_VFrac", "Pore_1", "CO2_Surf_m2/cm3", "dense", "Pore_3"]
    data = data.drop(['motif_furan', 'motif_pyrrole', 'motif_thiophene', 'motif_PO3'], axis=1)

    # Geometric
    if feature_set == 'geo': 
        Features = data[geom_features]
        print("\n\tFeature/Descriptor set:  Geometric {} features".format(Features.shape[1]))

    # Bag of Atoms + Geometric + APW-RDF
    elif feature_set == 'geo+rdf+boa':
        Features = data.drop(['wc', 'Unnamed: 0', 'Sel', 'label'], axis=1)
        Features = Features.drop(data.filter(like='motif'),axis=1)
        print("\n\tFeature/Descriptor set:  Geometric + APW-RDF + Bag of Atoms {} features".format(Features.shape[1]))

    # Bag of Atoms + APW-RDF
    elif feature_set == 'rdf+boa':
        Features = data.drop(['wc', 'Unnamed: 0', 'Sel', 'label'], axis=1)
        Features = Features.drop(geom_features, axis=1)
        Features = Features.drop(data.filter(like='motif'), axis=1)
        print("\n\tFeature/Descriptor set:  Bag of Atoms + APW-RDF {} features".format(Features.shape[1]))

    # Bag of Atoms + Geometric
    elif feature_set == 'geo+boa':
        Features = data.filter(like='epsilon')
        Features = pd.concat([Features,data.filter(like='sigma')], axis=1)
        Features = pd.concat([Features,data[geom_features]], axis=1)
        print("\n\tFeature/Descriptor set:  Bag of Atoms + Geometric {} features".format(Features.shape[1]))
        
    # Bag of Atoms only
    elif feature_set == 'boa':
        Features = data.filter(like='epsilon')
        Features = pd.concat([Features,data.filter(like='sigma')], axis=1)
        print("\n\tFeature/Descriptor set:  Bag of Atoms {} features".format(Features.shape[1]))
        
    # APW-RDF only
    elif feature_set == 'rdf':
        Features = data.filter(like='RDF')
        print("\n\tFeature/Descriptor set:  APW-RDF {} features".format(Features.shape[1]))
        
    # APW-RDF + Geometric
    elif feature_set == 'geo+rdf':
        Features = data.filter(like='RDF')
        Features = pd.concat([Features,data[geom_features]], axis=1)
        print("\n\tFeature/Descriptor set:  Geometric + APW-RDF {} features".format(Features.shape[1]))

    # Chemical Motifs only
    elif feature_set == 'mot':
        Features = data.filter(like='motif')
        print("\n\tFeature/Descriptor set: Chemical Motifs {} features".format(Features.shape[1]))

    # Chemical Motifs + Geometric
    elif feature_set == 'geo+mot':
        Features = data.filter(like='motif')
        Features = pd.concat([Features,data[geom_features]], axis=1)
        print("\n\tFeature/Descriptor set: Chemical Motifs + Geometric {} features".format(Features.shape[1]))
        
    # Chemical Motifs + Geometric + Bag of Atoms
    elif feature_set == 'geo+mot+boa':
        Features = data.filter(like='motif')
        Features = pd.concat([Features,data[geom_features]], axis=1)
        Features = pd.concat([Features,data.filter(like='sigma')], axis=1)
        Features = pd.concat([Features,data.filter(like='epsilon')], axis=1)
        print("\n\tFeature/Descriptor set: Chemical Motifs + Geometric + Bag of Atoms {} features".format(Features.shape[1]))
        
    # Chemical Motifs + Geometric + RDF
    elif feature_set == 'geo+mot+rdf':
        Features = data.filter(like='motif')
        Features = pd.concat([Features,data[geom_features]], axis=1)
        Features = pd.concat([Features,data.filter(like='RDF')], axis=1)
        print("\n\tFeature/Descriptor set: Chemical Motifs + Geometric + APW-RDF {} features".format(Features.shape[1]))

    # No valid feature set
    else:
        print("\n\tInvalid Feature_Set defined")
        sys.exit()

    return Features


####################User needs to define these parameters#######################

# The feature set you would like to use. Acceptable options include:
# geo: Geometric
# rdf: AP-RDFs
# mot: Chemical Motifs
# boa: Bag-of-Atoms
# rdf+boa: AP-RDFs and Bag-of-Atoms
# geo+rdf: Geometric and AP-RDFs
# geo+mot: Geometric and Chemical Motifs
# geo+boa: Geometric and Bag-of-Atoms
# geo+mot+boa: Geometric and Chemical Motifs and Bag-of-Atoms
# geo+rdf+boa: Geometric and AP-RDF and Bag-of-Atoms
# geo+mot+rdf: Geometric and Chemical Motifs and AP-RDF
Feature_Set = 'rdf+boa'

# Target you want to predict. Acceptable options include:
# wc: CO2 Working Capacity
# Sel: CO2/N2 Selectivity (note: the 's' is capitalized)
target = 'wc'

# Name of the csv file containing the descriptors
descriptor_csv = 'New_Clean_Stats_3.csv'

################################################################################

#ignore warning about .as_matrix() discontinuation in future versions
warnings.filterwarnings("ignore")

start_time = datetime.now()
print("Start: ",start_time.strftime("%c"))

# If CUDA device is available, then use it, otherwise use CPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("Device: ", device)

# Process input
print("\n\tReading in data... This may take a few minutes depending on your device and the size of your CSV file.")
data = pd.read_csv('{}/{}'.format(os.path.dirname(os.path.realpath(__file__)), descriptor_csv))
try:
    MOFs = data['Unnamed: 0']
except KeyError:
    pass

# Get the descriptors according to the desired Feature_Set
Features = get_features(Feature_Set, data)
# Scale the features using StandardScaler
scaler = StandardScaler().fit(Features)
Features = scaler.transform(Features)

# Convert arrays of data to tensors
Features = torch.from_numpy(Features).float()
Features = Features.to(device)

# Load the model corresponding to given target and descriptor set
print("\n\tLoading in PyTorch model...")
if target == 'wc':
    model = torch.load('{}/wc/{}_wc_model.pt'.format(os.path.dirname(os.path.realpath(__file__)),Feature_Set), map_location=torch.device(device))
    results_filename = 'CO2WorkingCapacityPredictions.csv'
elif target == 'Sel':
    model = torch.load('{}/Sel/{}_Sel_model.pt'.format(os.path.dirname(os.path.realpath(__file__)),Feature_Set), map_location=torch.device(device))
    results_filename = 'CO2N2SelectivityPredictions.csv'
else:
    print("Invalid selection of target property... exiting.")
    sys.exit()
model.eval()

print("\n\tMaking predictions on the dataset...")
y_predict = model(Features)
y_predict = y_predict.to('cpu').detach().numpy()
y_predict = [val for sublist in y_predict for val in sublist]
results = pd.DataFrame()
results['Predictions'] = y_predict

try:
    results.index = [MOFs]
except:
    pass

print("\n\tPreparing the CSV file with results...")
results.to_csv('{}/{}'.format(os.path.dirname(os.path.realpath(__file__)), results_filename))
print("\nSuccessful termination.")
end_time = datetime.now()
elapsed_time = end_time - start_time
print("End:",end_time.strftime("%c"))
print("Total time: {0:.1f} s".format(elapsed_time.total_seconds()))

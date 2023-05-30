# Zachary Webel
# zow2@georgetown.edu

'''
Step 4 is where I create actual data a nn can learn from. I used a window size of 30 amino acids and
create three tabular feature vectors, one for the desired properties, the current window properties and
the difference of what the model has built to what the desired properties are
'''
#########################################################################################


# Imports , some not needed
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from random import randint
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random


# Import data
embedding_path = 'aa_embeddings.csv'
embedding_matrix = pd.read_csv(embedding_path)
#Define column names
embedding_matrix.columns=['amino_acid','x1','x2','x3']


# import train and test seqs
train_path = 'train_database.csv'
test_path = 'test_database.csv' 
val_path = 'validation_database.csv'

train = pd.read_csv(train_path)
validation = pd.read_csv(val_path)
test = pd.read_csv(test_path)


# Define amino acids vector, sequence length and step size
# Since I tokenized each amino acid, step size will be 1 but you
# can use k-mer approach, just would need to create new embeddings
amino_acid_list = list(embedding_matrix['amino_acid'])
seq_length = 30
step_size = 1


# Initialize storage vectors
# storing the input and outputs
input_ = []
output = []
database = []
property_matrix = []


# Format data for model training and inference
'''
SUMMARY

for each sequence we...

1. Create a list of each amino acid in order, random replacement for X

2. Create a string representation of the sequence after random replacement

3. Calculate the TARGET FEATURES (end molecular properties we want the sequence to have)
in the training,val,testing case we just use the end features of the actual sequence, but
during generation you can define these however you want.

4. Loop through from the start position all the way until we reach length - 30 because we need to have 30 amino acids as the sequence.
This defines our window of 30 aa.

5. Define the current itteration input sequence as a list of 30 amino acids from position i to position i+30. Literally just a list of 30 aa

6. Define the output amino acid as the i+30 position

7. Create WINDOW FEATURES, These are the molecular properties from just the 30 amino acid window sequence

8. Define the current built sequence as everything in the sequence before i+30 position. So each itteration through the loop this
current built sequence gets bigger by one amino acid

9. Calculate the molecular properties of the current built sequence

10. Create TARGET CURRENT DIFFERENCE features. This is defined as target - current built for each feature. Just the signed difference.

11. IMPORTANT**** create a 2d list and append it to our input_ vector. This list translates each amino acid in the 30 aa window
into it's embedding vector. Just a normal translation.

12. One hot encode our output. We define the output vector as size 21 (20 amino acids + 1 stop tag). If the position
of the output amino acid is the last in the sequence we ignore the amino acid and append the stop tag so we don't duplicate the sequence

13. Concat the target, window, difference features into one vector and append it to the property matrix.
'''

# loop through each sequence
for sequence in validation['sequence']:

    # define a list of all aa with random replacements for invalid position then create string representation
    sequence_amino_acids = [aa if aa in amino_acid_list else random.choice(amino_acid_list) for aa in sequence]
    sequence = ''.join(sequence_amino_acids)
    
    # TARGET FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################################################
    analysed_seq = ProteinAnalysis(Seq(sequence))
    target_mw = analysed_seq.molecular_weight()
    target_pi = analysed_seq.isoelectric_point()
    ec = analysed_seq.molar_extinction_coefficient()
    target_ec_1, target_ec_2 = ec[0],ec[1]
    target_instability_index = analysed_seq.instability_index()
    target_aromaticity = analysed_seq.aromaticity()
    target_gravy = analysed_seq.gravy()
    target_isoelectric_point = analysed_seq.isoelectric_point()
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
    target_helix, target_turn, target_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
    target_features = [target_mw,target_pi,target_ec_1,target_ec_2,target_instability_index,target_aromaticity,target_gravy,target_isoelectric_point,target_helix,target_turn,target_sheet]
    ###################################################################################
    
    # loop through the seq aa list until you get to a point where there are no 30 aa window's left
    for i in range(0,len(sequence_amino_acids) - seq_length, step_size):
        
        # define the input sequence list as a 30 aa window from i to i+seq length
        input_sequence = sequence_amino_acids[i: i + seq_length]
        
        # output is the i+ seq length position
        output_amino_acid = sequence_amino_acids[i+seq_length]

        # Don't duplicate input-output pairs
        if(''.join(input_sequence)+output_amino_acid not in database):
            database.append(''.join(input_sequence)+output_amino_acid)
            
            
            
            # WINDOW FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
            ###################################################################################
            analysed_seq = ProteinAnalysis(Seq("".join(input_sequence)))
            window_mw = analysed_seq.molecular_weight()
            window_pi = analysed_seq.isoelectric_point()
            ec = analysed_seq.molar_extinction_coefficient()
            window_ec_1, window_ec_2 = ec[0],ec[1]
            window_instability_index = analysed_seq.instability_index()
            window_aromaticity = analysed_seq.aromaticity()
            window_gravy = analysed_seq.gravy()
            window_isoelectric_point = analysed_seq.isoelectric_point()
            secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
            window_helix, window_turn, window_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
            window_features = [window_mw,window_pi,window_ec_1,window_ec_2,window_instability_index,window_aromaticity,window_gravy,window_isoelectric_point,window_helix,window_turn,window_sheet]
            ###################################################################################
            
            
            
            # Create current built seq
            current_built_sequence = sequence_amino_acids[0: i + seq_length]
            current_built_sequence = ''.join(current_built_sequence)
            
            
            # CURRENT_BUILT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
            ###################################################################################
            analysed_seq = ProteinAnalysis(Seq(current_built_sequence))
            current_built_mw = analysed_seq.molecular_weight()
            current_built_pi = analysed_seq.isoelectric_point()
            ec = analysed_seq.molar_extinction_coefficient()
            current_built_ec_1, current_built_ec_2 = ec[0],ec[1]
            current_built_instability_index = analysed_seq.instability_index()
            current_built_aromaticity = analysed_seq.aromaticity()
            current_built_gravy = analysed_seq.gravy()
            current_built_isoelectric_point = analysed_seq.isoelectric_point()
            secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
            current_built_helix, current_built_turn, current_built_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
            current_built_features = [current_built_mw,current_built_pi,current_built_ec_1,current_built_ec_2,current_built_instability_index,current_built_aromaticity,current_built_gravy,current_built_isoelectric_point,current_built_helix,current_built_turn,current_built_sheet]
            target_current_difference = [target_features[i]-current_built_features[i] for i in range(len(target_features))]
            ###################################################################################


            # amino acid translation. For each aa in the input sequence append a list of the embedding vector for that aa by look up
            # in the embedding data frame
            input_.append([embedding_matrix.loc[embedding_matrix['amino_acid'] == aa].values[0][1:] for aa in input_sequence])


            # one hot with 21 possible outcomes
            # 20 aa and 1 stop tag
            # The index for the stop tag is the last position
            # The index for the amino acids follows the order I used for the embedding matrix
            one_hot_output = [0]*21
            if(i+seq_length+1==len(sequence_amino_acids)):
                one_hot_output[-1] = 1
            else:
                index = embedding_matrix[embedding_matrix['amino_acid'] == output_amino_acid].index[0]
                one_hot_output[index] = 1
            output.append(one_hot_output)
            
            # concat all feature vectors into one and append to the property matrix
            property_matrix.append(target_features+window_features+target_current_difference)



# Change to a dataframe
property_matrix = pd.DataFrame(property_matrix)            
            
# reshape to a tensor size (len, 30, 3)
X = np.reshape(input_[:(len(output))], (len(output), seq_length,3))
y = np.array(output)

# Force sequences to a float and one hot to ints 
X = np.array(X).astype(float)
y = np.array(y).astype(int)

# save
np.save('validation_input.npy',X)
np.save('validation_output.npy',y)
property_matrix.to_csv('validation_property_matrix.csv',index=False)















            

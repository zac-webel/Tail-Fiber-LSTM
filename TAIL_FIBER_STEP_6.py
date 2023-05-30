# Zachary Webel
# zow2@georgetown.edu

'''
Step 6 - Generation Script imports best model, and generates novel tail fiber sequences
'''
#########################################################################################


# Imports , some not needed
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from random import randint
from tensorflow.keras.layers import Dense, Flatten, LSTM, Concatenate, BatchNormalization,Input, Dropout
from tensorflow.keras.models import Model


# Import embeddings and sequences
embedding_matrix = pd.read_csv('aa_embeddings.csv')
embedding_matrix.columns=['amino_acid','x1','x2','x3']
amino_acid_list = list(embedding_matrix['amino_acid'])
action_space =  list(embedding_matrix['amino_acid'])
action_space.append('stop')
sequences = pd.read_csv('test_database.csv')

# Define seq len and step size
seq_length = 30
step_size = 1

# Import model
seq_input = Input(shape=(seq_length, 3))
prop_input = Input(shape=(33,))
lstm_1 = LSTM(3, return_sequences=True)(seq_input)
lstm_2 = LSTM(6, return_sequences=True)(lstm_1)
lstm_3 = LSTM(12, return_sequences=False)(lstm_2)
concatenation = Concatenate()([lstm_3, prop_input])
normalized = BatchNormalization()(concatenation)
dense = Dense(1000, activation='relu')(normalized)
dropout = Dropout(0.2)(dense)
dense = Dense(1000, activation='relu')(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1000, activation='relu')(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1000, activation='relu')(dropout)
dropout = Dropout(0.2)(dense)
dense = Dense(1000, activation='relu')(dropout)
dropout = Dropout(0.2)(dense)
output = Dense(21,activation='softmax')(dense)
model = Model(inputs=[seq_input,prop_input], outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.load_weights("best_model.h5")



# Define Target properties - this can be your choice or taken from any sequence
# In this case I'll use a random sequence from the test set
# Just make sure it is a valid seq so using random replacement if not

sequence = sequences.loc[21]['sequence']
sequence = ''.join([aa if aa in amino_acid_list else random.choice(amino_acid_list) for aa in sequence])

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




generated_properties = []
# create 50 sequences
for count in range(50):
    
    # RANDOM SEED
    generated_sequence = []
    for i in range(30):
        generated_sequence.append(amino_acid_list[randint(0,len(amino_acid_list)-1)])
        
    #generated_sequence = sequence_amino_acids[:30]
    sequence = sequences.loc[randint(0,len(sequences)-1)]['sequence']
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
    

    current_position = 0
    stop_flag = False
    while stop_flag == False:

        # window sequence
        input_sequence = generated_sequence[current_position: current_position + seq_length]
        model_input = []

        # sequence input
        model_input.append([embedding_matrix.loc[embedding_matrix['amino_acid'] == aa].values[0][1:] if aa in amino_acid_list else embedding_matrix.loc[embedding_matrix['amino_acid'] == amino_acid_list[randint(0,len(amino_acid_list)-1)]].values[0][1:] for aa in input_sequence])
        X = np.reshape(model_input[:1], (1, seq_length,3))
        X = np.array(X).astype(float)


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


        # CURRENT_BUILT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
        ###################################################################################
        current_built_sequence = "".join(generated_sequence)
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


        #property feature vector
        property_vector = [target_features+window_features+target_current_difference]
        property_vector = np.array(property_vector)

        # predict using the model
        predicted_distribution = model.predict([X,property_vector],verbose=0)
        # sample from the probabilities
        sampled_action = np.random.choice(action_space,p=predicted_distribution[0])
        # if stop is sampled... stop 
        if(sampled_action=='stop'):
            stop_flag = True
        else: # keep generating
            generated_sequence = generated_sequence + [sampled_action]
        current_position = current_position + 1


    # string representation of our created sequence
    protein = ""
    for amino_acid in generated_sequence:
        protein = protein + amino_acid
    # generated FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################################################
    analysed_seq = ProteinAnalysis(Seq(protein))
    generated_mw = analysed_seq.molecular_weight()
    generated_pi = analysed_seq.isoelectric_point()
    ec = analysed_seq.molar_extinction_coefficient()
    generated_ec_1, generated_ec_2 = ec[0],ec[1]
    generated_instability_index = analysed_seq.instability_index()
    generated_aromaticity = analysed_seq.aromaticity()
    generated_gravy = analysed_seq.gravy()
    generated_isoelectric_point = analysed_seq.isoelectric_point()
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
    generated_helix, generated_turn, generated_sheet = secondary_structure_fraction[0],secondary_structure_fraction[1],secondary_structure_fraction[2]
    generated_features = [generated_mw,generated_pi,generated_ec_1,generated_ec_2,generated_instability_index,generated_aromaticity,generated_gravy,generated_isoelectric_point,generated_helix,generated_turn,generated_sheet]
    generated_properties.append(generated_features)
    target_generated_difference = [(abs(target_features[feature]-generated_features[feature])) / (abs(target_features[feature]+generated_features[feature]) / 2) * 100 for feature in range(len(target_features))]
    ###################################################################################
    print(protein)
    print('Avg abs % diff: ', sum(target_generated_difference)/len(target_generated_difference))
    for difference in target_generated_difference:
        print(difference)
    print('\n \n \n')













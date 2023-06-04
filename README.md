# Tail-Fiber-Sequential Language Model
This Project is an improved design from my HIDS-506 Artificial Intelligence final project submission. Every step in the project is posted.

<img width="500" alt="Screen Shot 2023-06-04 at 11 03 10 AM" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/33764b61-429a-4004-9085-b5b27e2bc9fd">

# Application
Using completely randomly generated seed sequences, my neural network and generation script is able to generate protein sequences that have the predicted function of viral tail fiber, virion attachment, binding, in adition to replicating molecular properties that phage tail fibers satisfy. 

# Example Workflow
I want to create a protein sequences with similar properties to a sequence I have already identified with a predicted function of virus tail fiber
ID: YP_007236083.1
Description: long tail fiber protein distal subunit [Yersinia phage phiR1-RT]
 
 SEED: MADLSRIKFKRSSVAGKRPLPADIAEGELA
 
  Desired Properties
<img width="639" alt="Picture6" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/b974a7ab-142a-4d4d-939b-1545c57a01dd">

Using the Generation Script, I input the seed sequence and the desired molecular properties. The script will generate as many sequences as you desire. The seed is highlighted 

<img width="686" alt="Picture7" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/1e98adf3-93b7-45ca-b3d5-d1b1e7c2717a">


 The best sequence (or any you desire) is then chosen by chosen critera.
 <img width="938" alt="Picture8" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/ef59edfb-9a4c-4f4d-ad90-477e3e9c8859">

 Using thrid party software (ProteInfer - Convolutional neural network that predicts the gene ontology terms associated with amino acid sequences) we can predict the function. 
<img width="878" alt="Picture9" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/660dd32c-ecf0-47ce-856d-b2069779d92f">

<img width="937" alt="Screen Shot 2023-06-04 at 11 53 21 AM" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/20e63718-7551-486d-aacf-fa1b0f8362b0">

 We can then use AlphaFold to predict the structure
https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/5fe0a843-bb81-4ac6-9d66-cec88e5b9986



<img width="418" alt="Picture1 0png" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/5b49c0c2-d0c9-43bb-9ca2-4575494517c6">





# How to run
Each step is posted, view code in order STEP_n.py


# Health Problem:
The results of the back and forth battle between bacteria and the antibiotic drug development industry is starting to change. Natural evolution causes bacteria to evolve against modern medicine while the drug discovery and development timeline has remained a process which takes a considerable amount of time. If changes are not made to the drug development process, “With a decrease in the discovery rate of novel antibiotics, this threatens to take humankind back to a “pre-antibiotic era” of clinical care” (Romero-Calle). The results of such will see human beings fall seriously ill due to a bacterial infection from something as simple as falling at the park. In short: “Antibiotic resistance is arguably the biggest current threat to global health” (Altamirano). A possible counter to the bacteria could be bacteriophage therapy. Phages are viruses that kill bacteria. Each phage has a specific type of bacteria it looks for. Once the virus latches onto the bacteria cell, it punctures a hole in the cell wall, injects its dna, reproduces on a large scale, kills the bacteria cell, escapes then restarts the whole process. 


# Project Goal:
This paper will discuss the implementation of artificial intelligence in the creation of novel tail fibers of bacteriophages. Tail fibers were the chosen structure of interest to study because they are responsible for the recognition of the bacteria target. A future problem in phage therapy could be that there are no known phages that will attack a specific bacteria that has infected a patient. I propose the use of a sequential language model to generate the protein structure of the tail fibers needed in hopes of genetically editing a captured phage to specifically target such bacteria. 



# Data:
All data collected and used is freely available. The data is collected from the protein database in the Entrez Molecular Sequence Database. The query used is ''(long tail fibre[All Fields] OR long tail fiber[All Fields]) AND phage[All Fields]’. The fields collected were: id, description, sequence, organism and taxonomy.
<img width="916" alt="Picture1" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/26543e89-95b3-41f7-b764-cc7ea193ad14">


# Data Cleaning
<img width="768" alt="Picture2" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/4ae0bfb6-474d-4212-acb6-537a8c794423">


# Amino Acid Embeddings
<img width="362" alt="Picture3" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/b6172b88-d204-4a65-b463-22dc949720a2">

# Model
<img width="956" alt="Screen Shot 2023-05-25 at 8 37 51 AM" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/1bcecae6-bc9b-4c32-bac2-910312c62266">

The model takes two inputs:
1. Amino Acid Sequence Data
2. Molecular Features


# Input 1 - Amino Acid Sequence Feature Creation
* Initialize the start index to 0 and the stop index to 30
* Set the input_sequence to train_sequence[start:stop]
* For each amino acid in the input sequence, translate to the corresponding embedding matrix
* Set the output_amino_acid variable to the stop amino acid
* One hot encode an output vector corresponding to the index position of the output amino acid in the embedding data frame
* Increment start and stop tags
* Repeat for all training sequences
* Reshape the input sequences to (len, 30, 3)

<img width="279" alt="Picture4" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/b87b6e54-344f-47a6-9162-12ab96823bfe">


# Input 2 - Molecular Properties
Molecular weight, Isoelectric point, molar extinction coefficients, instability index, aromaticity, gravy, secondary structure fractions
Calculated using Bio.SeqUtils.ProtParam ProteinAnalysis

1. Target features – Can be taken from any protein, what the user desires
2. Window features – The molecular features of the current window (30 amino acids)
3. Difference Features – For each molecular feature: Target – Current built sequence feature
4. Concatenate the vectors 



# Reliability Curve
<img width="614" alt="Picture5" src="https://github.com/zac-webel/Tail-Fiber-LSTM-/assets/118777665/413359f3-55d8-4bad-95f3-898d9400e988">


Romero-Calle, Danitza et al. “Bacteriophages as Alternatives to Antibiotics in Clinical Care.” Antibiotics (Basel, Switzerland) vol. 8,3 138. 4 Sep. 2019, doi:10.3390/antibiotics8030138
 
 Kurzgesagt – In a Nutshell, director. The Deadliest Being on Planet Earth – The Bacteriophage. YouTube, YouTube, 13 May 2018, https://www.youtube.com/watch?v=YI3tsmFsrOg. Accessed 7 Apr. 2023. 
 
 Taslem Mourosi J, Awe A, Guo W, Batra H, Ganesh H, Wu X, Zhu J. Understanding Bacteriophage Tail Fiber Interaction with Host Surface Receptor: The Key "Blueprint" for Reprogramming Phage Host Range. Int J Mol Sci. 2022 Oct 12;23(20):12146. doi: 10.3390/ijms232012146. PMID: 36292999; PMCID: PMC9603124.

# zow2@georgetown.edu







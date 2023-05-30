# Zachary Webel
# zow2@georgetown.edu

'''
Step 5 - Creating, training and evauluating the model
'''
#########################################################################################


# Imports , some not needed
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, LSTM, Concatenate, BatchNormalization,Input, Dropout,Conv2D,Attention, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from keras import optimizers


# Define input shapes
seq_length = 30
step_size = 1
seq_input = Input(shape=(seq_length, 3))
prop_input = Input(shape=(33,))

# Model Architecture I made
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

# Create and compile model
model = Model(inputs=[seq_input,prop_input], outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

# Define early stopping creitera with a patience of 30
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True,save_weights_only=True ,monitor='val_loss', mode='min', verbose=1)
callbacks = [early_stop, model_checkpoint]

# Fit model
model.fit([X_train,prop_train],y_train,
         epochs=1000,
         batch_size=32,
         validation_data=[[X_val,prop_val],y_val],
         callbacks=callbacks)



# make reliability curve
predicted_distributions = model.predict([X_test,prop_test])
probability = []
result = []
# store the rouned probability and the binary result of next amino acid
for i in range(len(predicted_distributions)):
    for j in range(len(predicted_distributions[0])):
        probability.append(round(100*predicted_distributions[i][j],0))
        result.append(y_test[i][j])

# make a pandas df of the probs and binary result
import pandas as pd
test_predictions = pd.DataFrame()
test_predictions['probability'] = probability
test_predictions['result'] = result


x = []
y = []
cur = 0
# while loop through each valid probability 0-100
while cur<=1000:
    # if the model predicted an amino acid with that probability
    # store the probability and the actual frequency of that amino acid being a correct prediction
    if(len(test_predictions.loc[test_predictions.probability==cur])>100):
        x.append(cur)
        y.append(100*sum(test_predictions.loc[test_predictions.probability==cur]['result'])/len(test_predictions.loc[test_predictions.probability==cur])) 
    cur = cur + 1

# plot y=x and our threshold probabilities vs actual rate
import matplotlib.pyplot as plt
plt.scatter(x=x,y=y)
line = [x for x in range(0,100)]
plt.plot(line)


# Print test loss and accuracy
test_loss, test_accuracy = model.evaluate([X_test,prop_test],y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    #Exception handling when the length of input series is shorter than window_size
    if(len(series) <= window_size):
        X.append(series[:])
        return X, y
    
    #Append to X the sublist of series from index i to index i + window_size (excluding)
    #Append to y the item in series at index i + window_size
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i+window_size])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    #a LSTM layer with 5 nodes
    model.add(LSTM(5, input_shape=(window_size, 1)))
    #a fully connected module with one unit
    model.add(Dense(1))

    return model



### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unqchr=list(set(text))
    #print(unqchr)

    # remove as many non-english characters and character sequences as you can 
    unqchr = [c for c in unqchr if c in ['%', 'à', '&', '/', '*', 'è', '$', 'é', 'â', '@', "'", '"', '(', ')', '-', '1', '2','3','4','5','6','7','8','9','0']]
    for c in unqchr:
        text = text.replace(c, ' ')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    #iterate through the input text by step_size
    #Append to inputs the sublist of text from index i to index i + window_size (excluding)
    #Append to outputs the item in series at index i + window_size    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i+window_size])
    
    return inputs,outputs

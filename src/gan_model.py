#Importing necessary libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, TimeDistributed, Input
from keras.optimizers import Adam, SGD
import os

# Ensure output directory exists
if not os.path.exists(os.getcwd() + '/output/'):
    os.makedirs(os.getcwd() + '/output/')
    
# Shuffling function for 3D arrays
def shuffle3D(arr):
    for a in arr:
        np.random.shuffle(a)

# Add time steps to features
def dimX(x, ts):
    x = np.asarray(x)
    newX = []
    for i, c in enumerate(x):
        newX.append([c] * ts)
    return np.array(newX)

# Add time steps to target
def dimY(Y, ts, chars, char_idx):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool_)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)

# Sequence prediction with argmax
def prediction(preds):
    y_pred = []
    for i, c in enumerate(preds):
        y_pred.append([np.argmax(j) for j in c])
    return np.array(y_pred)

# Sequence to text conversion
def seq_txt(y_pred, idx_char):
    newY = []
    for i, c in enumerate(y_pred):
        newY.append([idx_char[j] for j in c])
    return np.array(newY)

# Convert SMILES output from the model
def smiles_output(s):
    smiles = []
    for i in s:
        smiles.append(''.join(str(k) for k in i))
    return smiles

# Generator model
def Gen(x_dash, y_dash):
    G = Sequential()
    # Define input shape using Input layer to avoid the warning
    G.add(Input(shape=(x_dash.shape[1], x_dash.shape[2])))  # Input layer
    G.add(TimeDistributed(Dense(x_dash.shape[2])))
    G.add(LSTM(216, return_sequences=True))
    G.add(Dropout(0.3))
    G.add(LSTM(216, return_sequences=True))
    G.add(Dropout(0.3))
    G.add(LSTM(216, return_sequences=True))
    G.add(TimeDistributed(Dense(y_dash.shape[2], activation='softmax')))
    G.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=2e-4))
    return G

# Discriminator model
def Dis(y_dash):
    D = Sequential()
    # Define input shape using Input layer to avoid the warning
    D.add(Input(shape=(y_dash.shape[1], y_dash.shape[2])))  # Input layer
    D.add(TimeDistributed(Dense(y_dash.shape[2])))
    D.add(LSTM(216, return_sequences=True))
    D.add(Dropout(0.3))
    D.add(LSTM(60, return_sequences=True))
    D.add(Flatten())
    D.add(Dense(1, activation='sigmoid'))
    D.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001))
    return D

# GAN model
def Gan(G, D):
    GAN = Sequential()
    GAN.add(G)  # Add Generator
    D.trainable = False
    GAN.add(D)  # Add Discriminator
    GAN.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=2e-4))
    return GAN

# Train Discriminator
def trainDis(D, data, x_dash, y_dash, mc=None):
    if mc is None:
        fake_data = G.predict(x_dash)
        targets = np.zeros(x_dash.shape[0]).astype(int)
        Dloss = D.fit(fake_data, targets, epochs=1, batch_size=32)
    else:
        fake_ydata = np.copy(y_dash)
        shuffle3D(fake_ydata)
        targets = np.zeros(x_dash.shape[0]).astype(int)
        Dloss = D.fit(fake_ydata, targets, epochs=1, batch_size=32)
    
    return Dloss.history['loss'][0]

# Train GAN
def trainGAN(GAN, x_dash):
    target = np.ones(x_dash.shape[0]).astype(int)
    gan_loss = GAN.fit(x_dash, target, epochs=1, batch_size=32)
    return gan_loss.history['loss'][0]

##read csv file
data = pd.read_csv("Data/Smiles_data.csv")
data = data.sample(frac=1).reset_index(drop=True)

Y=data.SMILES
Y.head()

# Check for NaN values and handle them (you can choose to drop or fill them)
data = data.dropna(subset=['SMILES'])  # Drop rows where 'SMILES' is NaN

# Ensure that X columns are numeric and handle errors during conversion
X = data.iloc[:, 1:7]

# Check if any values are non-numeric or NaN
# Optionally fill NaNs with a specific value like 0 or the column mean
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Now you can safely work with X
print(X.head())

# Padding SMILES to equal length
maxY = Y.str.len().max()
y = Y.str.ljust(maxY, fillchar='|')

# CharToIndex and IndexToChar functions
chars = sorted(set("".join(y.values.flatten())))
char_idx = {c: i for i, c in enumerate(chars)}
idx_char = {i: c for i, c in enumerate(chars)}

ts = y.str.len().max()
y_dash = dimY(y, ts, chars, char_idx)
x_dash = dimX(X, ts) 

# Initialize models
G = Gen(x_dash, y_dash)
D = Dis(y_dash)
GAN = Gan(G, D)

# Pre-training Discriminator
for i in range(20):
    shuffleData = np.random.permutation(y_dash)
    disloss = trainDis(D, shuffleData, x_dash, y_dash)
    print(f"Pre Training Discriminator Loss: {disloss}")
    

# Train GAN
episodes = 10
for episode in range(episodes):
    print(f"Epoch {episode}/{episodes}")
    disloss = trainDis(D, y_dash, x_dash, y_dash)
    disloss_mc = trainDis(D, y_dash, x_dash, y_dash, mc="mc")
    ganloss = trainGAN(GAN, x_dash)
    
    print(f"D loss={disloss} | D (mc) loss={disloss_mc} | GAN loss={ganloss}")
    
# Optionally save model weights
    if episode % 10 == 0:
        G.save(os.path.join(os.getcwd(), 'output', 'Gen.h5'))
        D.save(os.path.join(os.getcwd(), 'output', 'Dis.h5'))
        GAN.save(os.path.join(os.getcwd(), 'output', 'Gan.h5'))
        
# Sample predictions
    if episode % 100 == 0:
        print("Predicting Molecule")
        x_pred = [[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        x_pred = dimX(x_pred, ts)
        preds = G.predict(x_pred)
        y_pred = prediction(preds)
        y_pred = seq_txt(y_pred, idx_char)
        s = smiles_output(y_pred)
        print(s)
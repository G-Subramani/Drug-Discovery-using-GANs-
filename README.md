The project involves the use of SMILES strings (Simplified Molecular Input Line Entry System) to represent chemical compounds, and the goal appears to be generating new molecules (SMILES strings) based on the given input data.

### Key Points:

1. Data Overview: 
   - The dataset contains SMILES strings in the first column and several binary target columns (e.g., `cox2`, `estrogen`, `gelatinase`, `neuramidase`, `kinase`, `thrombin`) that likely represent the presence or absence of specific biological activity for each compound.
   - Each SMILES string corresponds to a chemical structure, and the target columns represent the labels for different types of biological activities.

2. Objective:
   - The project is training a Generative Adversarial Network (GAN) to learn the mapping from molecular data (SMILES strings) to biological activity labels.
   - Specifically, the GAN has a Generator that creates new SMILES strings (molecules), and a 
Discriminator that distinguishes between real and generated molecules.
   - The aim of the GAN is to generate plausible molecular structures (SMILES strings) that could exhibit the same biological activity patterns (as indicated in the target columns).

3. Key Steps in the Process:
   - Preprocessing SMILES Data: The SMILES strings are padded to the same length, and a character-to-index and index-to-character mapping is created for the SMILES string's characters.
   - Model Definitions:
     - The Generator (G) is a sequential model with LSTM layers that generates new molecular structures (SMILES strings).
     - The Discriminator (D) is a model that evaluates the validity of the generated SMILES strings by checking if they are similar to real molecules in terms of biological activity.
     - The GAN model combines the Generator and Discriminator for adversarial training.
   
4. Training Procedure:
   - The training process alternates between training the Discriminator on both real and fake data and training the Generator to fool the Discriminator into classifying generated SMILES strings as real.
   - Loss Functions: The Discriminator uses binary cross-entropy loss to distinguish between real and fake data, while the Generator is trained to minimize the Discriminator's ability to differentiate between real and fake data.

5. Post-Training Predictions: After the training process, the model generates new SMILES strings by using the trained Generator to predict new molecules based on some input data.

### Conclusion:
The purpose of the project is to use a GAN-based model to generate new molecular structures (SMILES strings) that exhibit similar biological activities to the real molecules in the dataset. The model learns the underlying patterns in the chemical structures and biological activity labels and generates new candidates that could potentially have similar properties.

This approach could be useful for drug discovery or chemical design, where the goal is to generate novel compounds that exhibit desired biological activities (e.g., targeting specific diseases or conditions).

# Drug-Discovery-using-GANs-

'''bash 
Project repo:https://github.com/
## step 1-Ctreate conda environment after opening the repository 
'''bash 
conda create -n drds python=3.8 -y

'''bash 
conda activate drds
'''

install the requirements 
'''bash 
pip install -r requirements.txt
'''

# SMILES GAN
This project implements a GAN to generate chemical compound representations (SMILES).

## Dependencies
- Python 3.9+
- Keras, Pandas, NumPy, Matplotlib, etc.

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the dataset in the `data/` folder.
4. Run the script: `python src/gan_model.py`.


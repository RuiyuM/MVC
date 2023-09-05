import pickle

# Specify the directory and file name
file_path = './log_AL/10patch_based_selection.pickle'

# Open the pickle file and load the data
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print the data
print(data)


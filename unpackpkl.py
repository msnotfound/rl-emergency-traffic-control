import pickle

file_path = 'vec_normalise.pkl' # Replace with the actual path to your PKL file

try:
    # Open the file in binary mode ('rb') and load the data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Now 'data' contains the deserialized Python object
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except pickle.UnpicklingError as e:
    print(f"Error unpickling the file: {e}")
except EOFError:
    print("Error: Reached the end of the file unexpectedly. The file might be incomplete or corrupted.")

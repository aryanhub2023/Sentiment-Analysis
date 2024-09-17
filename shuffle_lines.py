import random

def shuffle_lines_in_file(file_path):
    """
    Shuffle the lines in a text file and overwrite the file with the shuffled lines.

    Args:
    file_path (str): Path to the text file to be shuffled.
    """
    # Read the lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Shuffle the lines randomly
    random.shuffle(lines)
    
    # Write the shuffled lines back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# Define file paths
train_file_path = 'dataset/train.txt'
test_file_path = 'dataset/test.txt'

# Shuffle lines in both files
shuffle_lines_in_file(train_file_path)
shuffle_lines_in_file(test_file_path)

print("Lines in files shuffled successfully!")

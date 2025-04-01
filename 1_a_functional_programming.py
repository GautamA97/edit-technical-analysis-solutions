import numpy as np

def calculate_statistics(data_list):
    
    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std_dev = np.std(data_array, ddof=1)
    count = len(data_array)
    
    return mean, std_dev, count

# Example usage with the provided list
data = [5.99342831, 4.7234714, 6.29537708, 8.04605971, 4.53169325,4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009]

mean, std_dev, count = calculate_statistics(data)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
print(f"Count: {count}")
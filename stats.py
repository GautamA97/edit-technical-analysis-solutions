import numpy as np
import scipy.stats as stats

def calculate_statistics(data_list):
    
    data_array = np.array(data_list)
    mean = np.mean(data_array)
    std_dev = np.std(data_array, ddof=1)
    count = len(data_array)
    
    return mean, std_dev, count

def one_sample_z_test(data_list, threshold):

    mean, std_dev, n = calculate_statistics(data_list)
    
    z_stat = (mean - threshold) / (std_dev / np.sqrt(n))
    
    p_value = 1 - stats.norm.cdf(z_stat)
    
    is_bad_gene = mean > threshold
    
    return z_stat, p_value, is_bad_gene

test_data = [5.99342831, 4.7234714, 6.29537708, 8.04605971, 4.53169325,4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009]

threshold = 4
z_stat, p_value, is_bad = one_sample_z_test(test_data, threshold)
print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")
print(f"Is 'bad' gene: {is_bad}")
print(f"Interpretation: The sample mean {'does' if is_bad else 'does not'} exceed the threshold of {threshold}.")
print(f"The p-value of {p_value} {'indicates statistical significance' if p_value < 0.05 else 'does not indicate statistical significance'} at α=0.05.")
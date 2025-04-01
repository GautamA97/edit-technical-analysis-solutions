import pandas as pd
import numpy as np
import scipy.stats as stats

class GeneExpressionAnalyzer:
    def __init__(self, csv_file_path):
        
        self.data = pd.read_csv(csv_file_path, index_col=0)
        
        self.gene_names = self.data.index.tolist()
        self.patient_names = self.data.columns.tolist()
        
    def z_test_gene(self, gene_expression, threshold=4):

        gene_array = np.array(gene_expression)
        mean = np.mean(gene_array)
        std_dev = np.std(gene_array, ddof=1)
        n = len(gene_array)
        
        z_stat = (mean - threshold) / (std_dev / np.sqrt(n))
        
        p_value = 1 - stats.norm.cdf(z_stat)
        
        is_bad_gene = mean > threshold
        
        return z_stat, p_value, is_bad_gene, mean
    
    def analyze_all_genes(self, threshold=4):

        results = []
        
        for gene_name in self.gene_names:
            gene_expression = self.data.loc[gene_name].values
            z_stat, p_value, is_bad_gene, mean = self.z_test_gene(gene_expression, threshold)
            
            results.append({
                'gene_name': gene_name,
                'mean_expression': mean,
                'z_statistic': z_stat,
                'p_value': p_value,
                'classification': 'bad' if is_bad_gene else 'good'
            })
        
        return pd.DataFrame(results)
    
    
analyzer = GeneExpressionAnalyzer('technical_data/1_c_d.csv')
results = analyzer.analyze_all_genes(threshold=4)
print("\nGene Expression Analysis Results:")
print("=================================")
print(results.to_string())
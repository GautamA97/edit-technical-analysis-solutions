import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def plot_gene_correlation(self, gene1_name, gene2_name):

        gene1_expression = self.data.loc[gene1_name].values
        gene2_expression = self.data.loc[gene2_name].values
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=gene1_expression, y=gene2_expression)
        
        sns.regplot(x=gene1_expression, y=gene2_expression, scatter=False, line_kws={"color": "red"})
        
        correlation = np.corrcoef(gene1_expression, gene2_expression)[0, 1]
        
        plt.xlabel(f"{gene1_name} Expression")
        plt.ylabel(f"{gene2_name} Expression")
        plt.title(f"Correlation between {gene1_name} and {gene2_name} Expression\nCorrelation Coefficient: {correlation:.3f}")
        
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

if __name__ == "__main__":
    analyzer = GeneExpressionAnalyzer('technical_data/1_c_d.csv')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    analyzer.plot_gene_correlation('gene_0', 'gene_1')
    
    plt.subplot(1, 3, 2)
    analyzer.plot_gene_correlation('gene_2', 'gene_7')
    
    plt.subplot(1, 3, 3)
    analyzer.plot_gene_correlation('gene_3', 'gene_4')
    
    plt.tight_layout()
    plt.show()
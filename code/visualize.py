import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

sns.set()

datasets = {'cub': 'CUB200-2011',
           'mini-imagenet': 'mini-ImageNet',
           'cifar-fs': 'CIFAR-FS'}

line_styles = [':', '--', '-', '-.']

class ApproximateAccuracy(object):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        df["label"] = pd.to_numeric(df["label"])
        df = df[df["radius"] != -1]
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return ((df["label"] == df["predict"]) & (df["radius"] >= radius)).mean()
    
    def abstain_num(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return (len(df[df["predict"] == -1]) / len(df.index)) * 100

def plot_params(n, dataset, sigma, n_samples, alpha, line='-'):
        img_folder = "../data/certify/new/{dataset}/{n}shot/".format(dataset=dataset, n=n)  
        acc = ApproximateAccuracy(img_folder + "N{n_samples}_sigma{sigma}_alpha{alpha}.txt".format(sigma=sigma,
                                                                                               n_samples=n_samples,
                                                                                               alpha=alpha)) 
        print('{}shot, {} - abstain rate: '.format(n, dataset), acc.abstain_num(), '%')
        attack_radii = np.linspace(0.0, 0.5, 100)
        cert_radii = acc.at_radii(attack_radii)
        
        return attack_radii, cert_radii
        #plt.plot(attack_radii, cert_radii, line, linewidth=3, label='$\sigma$ = {}'.format(sigma))

def plot_sigma(n, dataset, sigma, n_samples, alpha, line='-'):
        attack_radii, cert_radii = plot_params(n, dataset, sigma, n_samples, alpha, line)
        plt.plot(attack_radii, cert_radii, line, linewidth=3, label='$\sigma$ = {}'.format(sigma))

def plot_n_samples(n, dataset, sigma, n_samples, alpha, line='-'):
        attack_radii, cert_radii = plot_params(n, dataset, sigma, n_samples, alpha, line)
        plt.plot(attack_radii, cert_radii, line, linewidth=3, label='n samples = {}'.format(n_samples))

def plot_alpha(n, dataset, sigma, n_samples, alpha, line='-'):
        attack_radii, cert_radii = plot_params(n, dataset, sigma, n_samples, alpha, line)
        plt.plot(attack_radii, cert_radii, line, linewidth=3, label='alpha = {}'.format(alpha))

# +
if __name__ == "__main__":
    sigmas = [0.25, 0.5, 1.0]
    alphas = [0.01, 0.001, 0.0001]
    
#     plt.figure(dpi=150)
#     for i, n_samples in enumerate([1000, 3000, 5000]):
#         plot_n_samples(1, dataset='cub200', sigma=1.0, n_samples=n_samples, alpha=0.001, line=line_styles[i])
#     plt.xlabel('Attack radius')
#     plt.ylabel('Certified accuracy')
#     plt.legend(prop={'size': 12})
#     plt.savefig('../images/1shot_CUB200-2011_diff_n.png')
    
    for n_shot in [1, 5]:
        for dataset in datasets.keys():
            plt.figure(dpi=150)
            for i, sigma in enumerate(sigmas):
                plot_sigma(n_shot, dataset=dataset, sigma=sigma, n_samples=100000, alpha=0.0001, line=line_styles[i])
            plt.xlabel('Attack radius', fontsize=15)
            plt.ylabel('Certified accuracy', fontsize=15)
            plt.legend(fontsize=15)
            plt.savefig('../images/new_{}shot_{}_sigmas.png'.format(n_shot, datasets[dataset]))
            
#             plt.figure(dpi=150)
#             for i, alpha in enumerate(alphas):
#                 plot_alpha(n_shot, dataset=dataset, sigma=1.0, n_samples=1000, alpha=alpha, line=line_styles[i])
#             plt.xlabel('Attack radius', fontsize=15)
#             plt.ylabel('Certified accuracy', fontsize=15)
#             plt.legend(fontsize=15)
#             plt.savefig('../images/{}shot_{}_aplhas.png'.format(n_shot, datasets[dataset]))

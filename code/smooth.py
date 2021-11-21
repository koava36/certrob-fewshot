import torch
import numpy as np
from math import ceil, pi

#import matplotlib.pyplot as plt
#from IPython.display import clear_output

EMB_SIZE = 512

class Smooth(object):
    """A smoothed classifier g """
    # to abstain, Smooth returns this int
    ABSTAIN = -1.0
        
    def __init__(self, base_model: torch.nn.Module, device: str, num_classes: int, sigma: float):
        self.base_model = base_model
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        
    
    def certified_radius(self, x: torch.tensor, true_centroid: torch.tensor, adv_centroid: torch.tensor):
        """Certified radius in l2-norm for sample x"""
        
        """
        :return: lower estimate of l2-norm of perturbation that doesn't change prediction on x
        """
        
        if len(true_centroid.shape) == 1:
            delta = self._adversarial_emb(x, true_centroid, adv_centroid)
            return abs(delta * self.sigma * np.sqrt(pi / 2))
        
        else:
            deltas = self._adversarial_emb_batch(x, true_centroid, adv_centroid)
            return abs(deltas * self.sigma * np.sqrt(pi / 2))
        
    
    def predict(self, args, x: torch.tensor, centroids: torch.tensor, centroid_classes: torch.tensor, show_bounds=False):
        """Define predicted by smoothed model class and adversarial class on sample x"""
        
        """
        :return: (predicted class, adversarial class, predicted class centroid, adversarial class centroid) or ABSTAIN
        """
        
        alpha = args.alpha
        m_values = args.M
        k_repeats = args.K
        n_samples = args.N
        batch_size = args.batch
#         def s_realizations(a, N):
#             coef = 1 / (N ** 2 - N)
#             pair_prods = a @ torch.transpose(a, -2, -1)
#             return coef * (torch.sum(pair_prods, axis=(-2, -1)) - torch.diagonal(pair_prods, dim1=-2, dim2=-1).sum(axis=-1))
        
        def s_realizations(a, N):
            step = 1000
            coef = 1 / N ** 2
            summ = 0
            for i in range(0, N, step):
                for j in range(N, 2 * N, step):
                    summ += torch.sum(a[j:j+step] @ torch.transpose(a[i:i+step], 0, 1))
            return summ * coef

        with torch.no_grad():
            
            sampled_embeddings = torch.empty((1, 2 * n_samples, EMB_SIZE))
            
            for k in range(k_repeats):
                # sample 2n values of f(x + eps) for each of m realization
                new_embeddings = self._sample_smoothed(x, m_values, 2 * n_samples, batch_size).cpu()
                sampled_embeddings = torch.cat((sampled_embeddings, new_embeddings))
                if k == 0:
                    sampled_embeddings = sampled_embeddings[1:]

                this_m_values = sampled_embeddings.shape[0]

                s_estimates = torch.empty((1, this_m_values)).to(self.device)

                #for each class, compute realizations of s_k = coef * sum(<f(x+eps_i) - c_k, f(x+eps_j) - c_k>)
                for i in range(self.num_classes):
                    dists = sampled_embeddings - centroids[i].cpu()

                    s = []
                    for j in range(this_m_values):
                        s.append(s_realizations(dists[j].to(self.device), n_samples))
                    s = torch.tensor(s, device=self.device)

                    s_estimates = torch.cat((s_estimates, s.unsqueeze(0)))
                s_estimates = s_estimates[1:]

                #confidence intervals for means of s_k for all classes
                conf_ints = self._confidence_intervals(s_estimates, alpha)

                # for debugging
#                 if show_bounds:
#                     clear_output(True)
#                     plt.figure(figsize=(18, 6))
#                     for i in range(conf_ints.shape[1]):
#                         plt.plot(conf_ints[:, i].cpu(), [0, 0], "o")
#                         plt.text(conf_ints[0, i], -0.05, '({a:.5f}, {b:.5f})'.format(a=conf_ints[0, i], b=conf_ints[1, i]))
#                     plt.xlim(conf_ints.cpu().min()-0.1, conf_ints.cpu().max()+0.1)
#                     plt.title('iteration: {k}'.format(k=k))
#                     plt.show()


                if self._robustness_condition(conf_ints):
                    return centroid_classes[torch.argsort(conf_ints[1, ...])[:2]], centroids[torch.argsort(conf_ints[1, ...])[:2]]
            
            return Smooth.ABSTAIN

        
    def _sample_smoothed(self, x: torch.tensor, m_values: int, n_samples: int, batch_size: int):
        """Sample values of g(x) Monte-Carlo estimation (without normalization)"""
        
        """
        :return: tensor of size [m_values, emb_size]
        """
        
        num = m_values * n_samples
        embeddings = torch.empty((1, EMB_SIZE)).to(self.device)
                                 
        with torch.no_grad():
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device=self.device) * self.sigma
                batch_emb = self.base_model(batch + noise).to(self.device)
                embeddings = torch.cat((embeddings, batch_emb))

            embeddings = embeddings[1:].reshape(m_values, n_samples, -1)
            
        return embeddings
    
    
    
    def _robustness_condition(self, conf_ints: torch.tensor):
        """
        Check if confidence intervals for the three smallest means don't intersect
        It means, the closest class always predicted with 1-alpha probability and there is an adversarial class
        """

        i_pred = torch.argmin(conf_ints[0, ...]).cpu()
        diff1 = conf_ints[1, i_pred] - conf_ints[0, np.r_[:i_pred, i_pred+1:self.num_classes]] # b_min - a_i, i != min
        cond1 =  torch.all(diff1 <= 0)
        
        ci_wo_pred = conf_ints[0, np.r_[:i_pred, i_pred+1:self.num_classes]]
        
        i_adv = torch.argmin(ci_wo_pred).cpu()
        i_adv = i_adv + (i_adv >= i_pred)
        diff2 = conf_ints[1, i_adv] - conf_ints[0, np.r_[:min(i_pred, i_adv), min(i_pred, i_adv) + 1:max(i_pred, i_adv), max(i_pred, i_adv) + 1:self.num_classes]]
        cond2 = torch.all(diff2 <= 0)
        
        return cond1 & cond2

    
    def _adversarial_emb(self, z: torch.tensor, x: torch.tensor, y: torch.tensor):
        return (0.5 * (torch.norm(y, p=2) ** 2 - torch.norm(x, p=2) ** 2) -  z @ (y - x)) / torch.norm(y - x, p=2)
    
    
    def _adversarial_emb_batch(self, z: torch.tensor, x: torch.tensor, y: torch.tensor):
        def norm_batch(a):
            return torch.diag(a @ torch.transpose(a, 0, 1))
        
        return (0.5 * (norm_batch(y) - norm_batch(x)) - torch.diag(z @ torch.transpose(y - x, 0, 1))) / norm_batch(y - x) 
               
        
    def _confidence_intervals(self, x: torch.tensor, alpha: float):
        """
        Confidence interval on sample's mean using Hoeffding's inequality 
        """
        x_means = x.mean(axis=1)
        t = np.sqrt((- np.log(alpha / 2) / (x.shape[1]) / 8))
        square_conf_ints = x_means.repeat(2, 1) + torch.tensor([[-t], [+t]]).to(self.device)
        square_conf_ints[square_conf_ints < 0] = 0                                                       
        conf_ints = torch.sqrt(square_conf_ints)
        
        return conf_ints

def divide_batch(batch, target, n_support):
    
    def supp_idxs(c):
        return torch.where(target==c)[0][:n_support]
    
    classes = torch.unique(target)
    n_classes = len(classes)
    
    n_query = len(torch.where(target==classes[0].item())[0]) - n_support
    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.vstack([batch[idx_list] for idx_list in support_idxs])
    support_target = torch.hstack([target[idx_list] for idx_list in support_idxs])
    
    query_idxs = torch.stack(list(map(lambda c: torch.where(target==c)[0][n_support:], classes))).view(-1)
    
    query_samples = batch[query_idxs]
    query_target = target[query_idxs]
    
    return support_samples, support_target, query_samples, query_target




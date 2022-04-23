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
        
    
    def predict(self, args, x: torch.tensor, centroids: torch.tensor, centroid_classes: torch.tensor):
        """Define predicted by smoothed model class and adversarial class on sample x
           Speed-up version
        """
        """
        :return: (predicted class, adversarial class, predicted class centroid, adversarial class centroid, n_samples) or ABSTAIN
        """
        
        alpha = args.alpha
        k_repeats = args.K
        n_samples = args.N
        batch_size = args.batch
        
        get_mean_quadr = lambda a, b: torch.mean(a, axis=0) @ torch.mean(torch.transpose(b, 1, 0), axis=1)
        get_mean_lin = lambda a: 2 * a @ torch.transpose(centroids, 1, 0)

        with torch.no_grad():
            
            mean_quadr = 0
            mean_lin = torch.zeros(self.num_classes, device=self.device)
            classes_const = torch.diag(centroids @ torch.transpose(centroids, 1, 0), 0)
            
            
            for k in range(k_repeats):
                # sample 2n values of f(x + eps) for each of m realization
                new_embeddings = self._sample_smoothed(x, 1, 2 * n_samples, batch_size)[0]

                #for each class, update coef * (<sumf(x+eps_i), sumf(x+eps_j)> and <sumf(x+eps_i)- c_k>
                
                #calculate conf bounds for mean <f(x+eps_i), f(x+eps_j)>
                new_mean_quadr = get_mean_quadr(new_embeddings[:n_samples, :], new_embeddings[n_samples:2*n_samples, :])

                #calculate conf bounds for <f(x+eps_i), c_k>
                new_mean_lin = get_mean_lin(torch.mean(new_embeddings, dim=0)) # [num_classes]
                
                mean_quadr = (k * mean_quadr + new_mean_quadr) / (k + 1)
                mean_lin = (k * mean_lin + new_mean_lin) / (k + 1)
                    

                #confidence intervals for means of s_k for all classes
                conf_ints_quadr = self._confidence_intervals(mean_quadr, (n_samples ** 2) * (k + 1), alpha, 4.)
                conf_ints_lin = self._confidence_intervals(mean_lin, n_samples * (k + 1), alpha, 4.)

                index = torch.LongTensor([1, 0])
                conf_ints = torch.sqrt(conf_ints_quadr - conf_ints_lin[index] + classes_const)


                if self._robustness_condition(conf_ints):
                    #print('dist to ptototypes: ', conf_ints)
                    return centroid_classes[torch.argsort(conf_ints[1, ...])[:2]], centroids[torch.argsort(conf_ints[1, ...])[:2]], n_samples * (k+1)
            
            #print('dist to ptototypes: ', conf_ints)
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
               
        
    def _confidence_intervals(self, x_means: torch.tensor, n: int, alpha: float, bound_const: float):
        """
        Confidence interval on sample's mean using Hoeffding's inequality 
        """
        t = np.sqrt((- np.log(alpha / 2) / (((bound_const ** 2) / 2) * n)))
        square_conf_ints = x_means.repeat(2, 1) + torch.tensor([[-t], [+t]]).to(self.device)

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




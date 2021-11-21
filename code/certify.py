import argparse
import torch
import os, sys
import numpy as np
import warnings
from time import time
import datetime
import gc
import torch
from smooth import divide_batch, Smooth
from parser_certify import get_parser
from protonet.train import init_protonet,init_seed, init_dataloader

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    
def get_centroids(num_support: int, support_embs: torch.tensor, support_target:torch.tensor):
    '''
    Return class centroids and corresponding targets for n-shot learning
    '''
    n = len(support_samples)
    assert n % num_support == 0, "Number of support samples must be divisible by number of support per class"
    
    if num_support == 1:
        # 1shot case
        centroids = support_embs
        return support_target
    else:
        # 5shot case
        n = len(support_samples)
        centroids = model(support_samples).reshape(n // num_support, num_support, -1)
        centroids = torch.nn.functional.normalize(centroids.mean(axis=1), p=2, dim=1)
        support_target = support_target[::num_support]
        
    return centroids, support_target

def predict_with_radius(args, model, sample: torch.tensor, centroids: torch.tensor, centroid_target: torch.tensor)
        '''
        Predict of smoothed model on sample (or abstain), certified radius on sample and time per sample
        '''
        before_time = time()
    
        pred = smoothed_model.predict(sample, centroids, centroid_target, args)
        
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        if pred != -1.0:
            pred_class, adv_class = pred[0][0].cpu().item(), pred[0][1].cpu().item()
            pred_centroid, adv_centroid = pred[1][0], pred[1][1]
            smoothed_embedding = smoothed_model._sample_smoothed(sample, m_values=1, n_samples=args.N, batch_size=args.batch).mean(dim=1)
            radius = smoothed_model.certified_radius(smoothed_embedding, pred_centroid, adv_centroid).cpu().item()
        else:
            pred_class = pred
            radius = -1.0
            
        return pred_class, radius, time_elapsed

    
    
if __name__ == "__main__":
    args = get_parser().parse_args()
    device = 'cuda:{n}'.format(n=args.cuda_number) if torch.cuda.is_available() and args.cuda else 'cpu'
    
    init_seed(args)
    test_dataloader = init_dataloader(args, 'test')
    model = init_protonet(args)
    model_path = os.path.join(args.base_classifier, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval();
    
    smoothed_model = Smooth(base_model=model, device=device, num_classes=args.classes_per_it_val, sigma=args.sigma)
    
    f = open(args.outfile, 'w')
    print("label\tpredict\tradius\ttime", file=f, flush=True)
    
    for j, batch in enumerate(test_dataloader):
        if j % args.skip != 0:
            continue
        if j == args.max:
            break
            
        x, y = batch[0].to(device), batch[1].to(device)
        support_samples, support_target, query_samples, query_target = divide_batch(x, y, n_support=args.num_support_val)

        centroids, support_taget = get_centroids(args.num_support_val, model(support_samples), support_target)
        
        for i in range(len(query_samples)):

            pred_class, redius, time_elapsed = predict_with_radius(args, model, query_samples[i, ...],
                                                                   centroids, support_target)
    
            print("{}\t{}\t{:.4}\t{}".format(query_target[i].cpu().item(), pred_class, radius, time_elapsed), file=f, flush=True)
                
    f.close()
    clear_cuda()
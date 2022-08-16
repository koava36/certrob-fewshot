import argparse
import torch
import os, sys
import numpy as np
import warnings
from time import time
import datetime
import gc
import torch

sys.path.append('protonet/')

from smooth import divide_batch, Smooth
from parser_certify import get_parser
from protonet.train import init_protonet, init_seed, init_dataloader

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def get_centroids(num_support, support_embs, support_target):
    '''
    Return class centroids and corresponding targets for n-shot learning
    '''
    n = len(support_embs)
    assert n % num_support == 0, "Number of support samples must be divisible by number of support per class"
    
    if num_support == 1:
        # 1shot case
        centroids = support_embs
        support_classes = support_target
    else:
        # 5shot case
        n = len(support_embs)
        centroids = support_embs.reshape(n // num_support, num_support, -1)
        centroids = torch.nn.functional.normalize(centroids.mean(axis=1), p=2, dim=1)
        support_classes = support_target[::num_support]
        
    return centroids, support_classes

def predict_with_radius(args, smoothed_model, sample: torch.tensor, centroids: torch.tensor, centroid_target: torch.tensor):
        '''
        Predict of smoothed model on sample (or abstain), certified radius on sample and time per sample
        '''
        before_time = time()
        
        pred = smoothed_model.predict(args, sample, centroids, centroid_target)
        
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        if pred != -1.0:
            pred_class, adv_class = pred[0][0].cpu().item(), pred[0][1].cpu().item()
            pred_centroid, adv_centroid = pred[1][0], pred[1][1]
            n_samples = pred[2]

            gamma_lcb, radius = smoothed_model.certified_radius(sample, pred_centroid, adv_centroid)
            gamma_lcb = gamma_lcb.cpu().item()
            radius = radius.item()

        else:
            pred_class = pred
            radius = -1.0
            gamma_lcb = 0.0
            n_samples = args.N * args.K
            
        return pred_class, gamma_lcb, radius, time_elapsed, n_samples

    

if __name__ == "__main__":
    args = get_parser().parse_args()
    device = 'cuda:{n}'.format(n=args.cuda_number) if torch.cuda.is_available() and args.cuda else 'cpu'
    
    init_seed(args)
    test_dataloader = init_dataloader(args, 'test')
    model = init_protonet(args)
    model_path = os.path.join(args.base_classifier, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
    model.eval();
    
    smoothed_model = Smooth(base_model=model, device=device, num_classes=args.classes_per_it_val, sigma=args.sigma, alpha=args.alpha, mode=args.mode)
    
    if args.descr:
        outfile = os.path.join(args.outdir, 'N{}_sigma{}_alpha{}_{}.txt'.format(args.N * args.K, args.sigma, args.alpha, args.descr))
    else:
        outfile = os.path.join(args.outdir, 'N{}_sigma{}_alpha{}.txt'.format(args.N * args.K, args.sigma, args.alpha))
        

    f = open(outfile, 'w')
    print("label\tpredict\tgamma LCB\tradius\ttime\tN samples", file=f, flush=True)
    next_ind = 0

    
    print("=== Certifying on {dataset}, {n_shot}-shot, with N = {N}, sigma = {sigma}, alpha = {alpha} ===".format(dataset=args.dataset,
                                                                                                   n_shot=args.num_support_val,
                                                                                                   N=args.N,
                                                                                                   sigma=args.sigma,
                                                                                                   alpha=args.alpha))
    if args.max < 0:
        num_images = args.num_query_val * args.classes_per_it_val * len(test_dataloader) // args.skip
    elif args.max >= 0:
        num_images = args.num_query_val * args.classes_per_it_val * args.max // args.skip
    processed_images = 0
    
    curr_ind = 0
    for j, batch in enumerate(test_dataloader):
        
        if j % args.skip != 0:
            continue
        if j == args.max:
            break   
        
        if curr_ind < next_ind:
            continue
            
        curr_ind += 1
        
        x, y = batch[0].to(device), batch[1].to(device)
        support_samples, support_target, query_samples, query_target = divide_batch(x, y, n_support=args.num_support_val)
        
        centroids, support_classes = get_centroids(args.num_support_val, model(support_samples), support_target)
        
        for i in range(len(query_samples)):
            processed_images += 1
            print('Image {} from {}'.format(processed_images, num_images), end="\r")
            
            pred_class, gamma_lcb, radius, time_elapsed, n_samples = predict_with_radius(args, smoothed_model, query_samples[i, ...],
                                                                   centroids, support_classes)
             
            print("{}\t{}\t{:.4}\t{:.4}\t{}\t{}".format(query_target[i].cpu().item(), pred_class, gamma_lcb, radius, time_elapsed, n_samples), file=f, flush=True)
                
    f.close()

import sys
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
import pickle
# filterwarnings("ignore")
from tqdm import tqdm

import argparse
import logging
from datetime import datetime
from pyHGT.utils import get_logger


parser = argparse.ArgumentParser(description='Training GNN on Paper-Field (L2) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--model_name', type=str, default='default',
                    help='Model name, for ablation study.')
parser.add_argument('--saved_model', type=str, default='default',
                    help='Path to saved model, for test.')
parser.add_argument('--task_name', type=str, default='PF',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_NN',
                    help='CS, Medicion or All: _CS or _Med or (empty)')         
parser.add_argument('--level', type=int, default=2,
                    help='paper field level: 1, 2, etc')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
parser.add_argument('--no_meta_rela', action="store_true", default=False,
                    help='Do not use meta relation parametrization')
parser.add_argument('--no_msg_rela', action="store_true", default=False,
                    help='Use the same attention weight for both message passing and attention')
'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=1,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--repeat', type=int, default=2,
                    help='How many time to train over a singe batch (reuse data)') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping') 
parser.add_argument('--test', action="store_true", default=False,
                    help='Only test and no training')

args = parser.parse_args()
today = datetime.now()
logger = get_logger(filename='out/log_{}_{}.log'.format(args.task_name + str(args.level) + args.model_name, today.strftime("%Y-%m-%d-%H:%M")), level=logging.DEBUG)

rev_PF_level = 'rev_PF_in_L' + str(args.level)
PF_in_level = 'PF_in_L' + str(args.level)

# logger = logging.get_logger()

logger.info(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
if args.model_name == 'default':
    save_path = os.path.join(args.model_dir, args.task_name + str(args.level) +  '_' + args.conv_name+ '_'+ today.strftime("%Y-%m-%d-%H:%M"))
else:
    save_path = os.path.join(args.model_dir, args.task_name + str(args.level) +  '_' + args.model_name + '_'+ today.strftime("%Y-%m-%d-%H:%M"))
    
    

logger.info (os.path.join(args.data_dir, 'graph%s.pk' % args.domain))
graph = renamed_load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
# with open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb') as file_obj:
#     graph = renamed_load(pickle.loads(file_obj.read()))


train_range = {t: True for t in graph.times if t != None and t < 2015 and t >=2014}
valid_range = {t: True for t in graph.times if t != None and t >= 2015  and t <= 2016}
test_range  = {t: True for t in graph.times if t != None and t > 2016}

types = graph.get_types()
'''
    cand_list stores all the L2 fields, which is the classification domain.
'''
cand_list = list(graph.edge_list['field']['paper'][PF_in_level].keys())
'''
Use KL Divergence here, since each paper can be associated with multiple fields.
Thus this task is a multi-label classification.
'''
criterion = nn.KLDivLoss(reduction='batchmean')

def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''

    # print("entering node_classification_sample")

    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace = False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, \
                inp = {'paper': np.array(target_info)}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width)



    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field'][rev_PF_level]:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['paper']['field'][rev_PF_level] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper'][PF_in_level]:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['field']['paper'][PF_in_level] = masked_edge_list
    
    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        for source_id in pairs[target_id][0]:
            ylabel[x_id][cand_list.index(source_id)] = 1
    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = np.arange(args.batch_size) + node_dict['paper'][0]

    # print("Done with node_classification_sample")
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel
    
def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in tqdm(np.arange(args.n_batch)):
        p = pool.apply_async(node_classification_sample, args=(randint(), \
            sel_train_pairs, train_range))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), \
            sel_valid_pairs, valid_range))
    jobs.append(p)
    return jobs

def prepare_data_no_pool():
    jobs = []
    for batch_id in tqdm(np.arange(args.n_batch)):
        result = node_classification_sample(randint(), sel_train_pairs, train_range)
        jobs.append(result)
    result = node_classification_sample(randint(), sel_valid_pairs, valid_range)
    jobs.append(result)
    return jobs




train_pairs = {}
valid_pairs = {}
test_pairs  = {}
'''
    Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
'''
for target_id in graph.edge_list['paper']['field'][rev_PF_level]:
    for source_id in graph.edge_list['paper']['field'][rev_PF_level][target_id]:
        _time = graph.edge_list['paper']['field'][rev_PF_level][target_id][source_id]
        if _time in train_range:
            if target_id not in train_pairs:
                train_pairs[target_id] = [[], _time]
            train_pairs[target_id][0] += [source_id]
        elif _time in valid_range:
            if target_id not in valid_pairs:
                valid_pairs[target_id] = [[], _time]
            valid_pairs[target_id][0] += [source_id]
        else:
            if target_id not in test_pairs:
                test_pairs[target_id]  = [[], _time]
            test_pairs[target_id][0]  += [source_id]


np.random.seed(43)
'''
    Only train and valid with a certain percentage of data, if necessary.
'''
sel_train_pairs = {p : train_pairs[p] for p in np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace = False)}
sel_valid_pairs = {p : valid_pairs[p] for p in np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace = False)}

            
'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
use_meta_rela = not args.no_meta_rela
use_msg_rela = not args.no_msg_rela
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper']['emb'].values[0]) + 401, \
          n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
          num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1, use_meta_rela=use_meta_rela, use_msg_rela=use_msg_rela).to(device)
classifier = Classifier(args.n_hid, len(cand_list)).to(device)

model = nn.Sequential(gnn, classifier)


if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters())
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters())
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val   = 0
train_step = 1500

# pool = mp.Pool(args.n_pool)
st = time.time()
# jobs = prepare_data(pool)

if args.test == False:
    jobs = prepare_data_no_pool()
    for epoch in tqdm(np.arange(args.n_epoch) + 1):
        '''
            Prepare Training and Validation Data
        '''
        train_data = [job for job in jobs[:-1]]
        # train_data = [job.get() for job in jobs[:-1]]

        # print("len train_data: ", len(train_data)) 
        # valid_data = jobs[-1].get()
        valid_data = jobs[-1]
        # pool.close()
        # pool.join()
        '''
            After the data is collected, close the pool and then reopen it.
        '''
        # pool = mp.Pool(args.n_pool)
        # jobs = prepare_data(pool)
        et = time.time()
        # print('Data Preparation: %.1fs' % (et - st))
        
        '''
            Train (time < 2015)
        '''
        model.train()
        train_losses = []
        torch.cuda.empty_cache()
        for _ in range(args.repeat):
            for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in tqdm(train_data):
                node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))
                res  = classifier.forward(node_rep[x_ids])

                # print("done forward")
                loss = criterion(res, torch.FloatTensor(ylabel).to(device))

                optimizer.zero_grad() 
                torch.cuda.empty_cache()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                train_losses += [loss.cpu().detach().tolist()]
                train_step += 1
                scheduler.step(train_step)
                del res, loss
        '''
            Valid (2015 <= time <= 2016)
        '''
        model.eval()
        with torch.no_grad():
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res  = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))
            
            '''
                Calculate Valid NDCG. Update the best model based on highest NDCG score.
            '''
            valid_res = []
            for ai, bi in zip(ylabel, res.argsort(descending = True)):
                valid_res += [ai[bi.cpu().numpy()]]
            valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
            
            
            if valid_ndcg > best_val:
                best_val = valid_ndcg
                torch.save(model, save_path)
                print('UPDATE!!!')
            
            st = time.time()
            logger.info(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
                (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                        loss.cpu().detach().tolist(), valid_ndcg))
            stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
            del res, loss
        del train_data, valid_data


    '''
        Evaluate the trained model via test set (time > 2016)
    '''

    with torch.no_grad():
        test_res = []
        for _ in range(10):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                        node_classification_sample(randint(), test_pairs, test_range)
            paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                        edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
            res = classifier.forward(paper_rep)
            for ai, bi in zip(ylabel, res.argsort(descending = True)):
                test_res += [ai[bi.cpu().numpy()]]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        logger.info('Last Test NDCG: %.4f' % np.average(test_ndcg))
        test_mrr = mean_reciprocal_rank(test_res)
        logger.info('Last Test MRR:  %.4f' % np.average(test_mrr))




if args.test:
    if args.saved_model != 'default':
        save_path = os.path.join(args.model_dir, args.saved_model)
    print("loading model from ", save_path)
best_model = torch.load(save_path).to(device)
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                    node_classification_sample(randint(), test_pairs, test_range)
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
        res = classifier.forward(paper_rep)
        for ai, bi in zip(ylabel, res.argsort(descending = True)):
            test_res += [ai[bi.cpu().numpy()]]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    logger.info('Best Test NDCG: %.4f' % np.average(test_ndcg))
    test_mrr = mean_reciprocal_rank(test_res)
    logger.info('Best Test MRR:  %.4f' % np.average(test_mrr))

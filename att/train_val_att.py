import torch.utils.data as data
import sys
sys.path.append("..")
sys.path.append(".")
from eval_metrics.eval import *


from amazon_men import *
from amazon_women import *
from pog import *
from pogfull import *
from model_att import *
import time
import yaml

def train(opt):
  def save_model(model, opt, bestndcg):
    name = time.time()
    torch.save(model.state_dict(), opt.save_path+ opt.dataset + '_' + str(opt.type)+'_'+str(opt.save_epoch) + '_' + str(bestndcg)[:6] + '_' + str(name)[:6])

  print('load training data...')
  if opt.dataset == 'women':
    data_train = WomenTrain(opt.data_path)
    data_test = WomenTest(opt.data_path)
  elif opt.dataset == 'men':
    data_train = MenTrain(opt.data_path)
    data_test = MenTest(opt.data_path)
  elif opt.dataset == 'pog':
    data_train = PogTrain(opt.data_path)
    data_test = PogTest(opt.data_path)
  elif opt.dataset == 'pog-full':
    data_train = PogfullTrain(opt.data_path)
    data_test = PogfullTest(opt.data_path)

  t1 = time.time()
  data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

  print('load test data...')
  data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=collate_val)
  print('data done.', time.time()-t1)

  opt.vocab_size = data_train.vocab_size
  opt.self_att = False
  opt.DEVICE = torch.torch.device("cuda:"+str(opt.gpu) if torch.cuda.is_available() else "cpu")

  print(opt)
  print('building model')
  model = Model_att(opt).to(opt.DEVICE)
  print('model built')
  iter_cnt = 0
  eval_freq = 10

  bestpre = 0
  bestndcg = 0
  bestap = 0
  bestmrr = 0
  bestrec = 0

  lr_update = model.lr
  for ep_id in range(opt.max_eps):
    t0 = time.time()
    loss_epoch = 0
    if ep_id % 200 == 0 and ep_id != 0:
      lr_update = lr_update*0.5
      model.optimizer = torch.optim.Adam(model.params, lr=lr_update, weight_decay = opt.weight_decay)
      print('change learning rate from {} to {}'.format(lr_update*2,lr_update))
    for iter_id, train_batch in enumerate(data_loader_train):
      model.train()
      model.train_step(*train_batch)
      iter_cnt += 1
      loss_epoch += model.loss.data
    t1 = time.time()
    print('Epoch: {}, total iter: {}, loss: {}, time: {}'.format(ep_id,iter_cnt,loss_epoch/(iter_id+1.), t1-t0))
    if ep_id%eval_freq==0:
      print('Eval...Epoch {}'.format(ep_id))
      rank_res = {}
      pre = []
      ndcg = []
      ap = []
      mrr = []
      rec = []
      for test_batch in data_loader_test:
        model.eval()
        scores = model.get_output(test_batch[0], test_batch[3])
        # attention_alpha = model.att_layer.alphas

        uid = test_batch[4]
        targets = test_batch[1]
        candidates = test_batch[2]
        _, sort_idx = torch.sort(scores,descending=True)
        rank_list = [candidates[i] for i in sort_idx]
        rank_res[uid] = rank_list
        metrics = compute_user_metrics(rank_list, targets, opt.topk)
        pre.append(metrics[0])
        rec.append(metrics[1])
        ap.append(metrics[2])
        ndcg.append(metrics[3])
        mrr.append(metrics[4])

      print('scores: Precision %.4f, Recall %.4f, mAP %.4f, NDCG %.4f, MRR %.4f'%(np.mean(pre),np.mean(rec),np.mean(ap),np.mean(ndcg),np.mean(mrr)))
      if np.mean(ndcg) > bestndcg:
        best_epoch = ep_id
        bestpre = np.mean(pre)
        bestrec = np.mean(rec)
        bestap = np.mean(ap)
        bestndcg = np.mean(ndcg)
        bestmrr = np.mean(mrr)
      t2 = time.time()
      print('best epoch %d: Precision %.4f, Recall %.4f, mAP %.4f, NDCG %.4f, MRR %.4f'%(best_epoch, bestpre, bestrec, bestap, bestndcg, bestmrr))
      print('evaluate time: {}'.format(t2-t1))
    if ep_id == opt.save_epoch:
      save_model(model, opt,bestndcg)
      print('the epoch {} model has been saved'.format(ep_id))

def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
      if isinstance(j, dict):
          setattr(top, i, obj_dic(j))
      elif isinstance(j, seqs):
          setattr(top, i,
              type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
      else:
          setattr(top, i, j)
    return top


def main():
  opt = yaml.load(open("./config_att.yaml"))
  opt = obj_dic(opt)

  if opt.dataset in ['men', 'women']:
      opt.num_node = 7
      opt.data_path = '/home/joan/djj_mask/data/amazon-' + opt.dataset + '-group-cp_mask'
  elif opt.dataset == 'pog':
      opt.num_node = 6
      opt.data_path = '~/djj_mask/data/pog'

  elif opt.dataset == 'pog-full':
      opt.num_node = 25
      opt.data_path = '~/djj_mask/data/pog-full'

  train(opt)

if __name__ == '__main__':
    np.random.seed(2016)
    torch.manual_seed(2016)
    torch.cuda.manual_seed_all(2016)
    main()



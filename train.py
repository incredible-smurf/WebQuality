#/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import os
import math
import argparse
from tqdm import tqdm
import time

from utils import loggerfunc, read_data_path, MyDataset
from models.UniQuality import UniQualityModel
from metric import PRFMetric


def train(args, train_dataloader, model, test_dataloader=None):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lossfunc = nn.CrossEntropyLoss()
    metric = PRFMetric(args.num_classes, args.label_type)
    dev_best_f1 = float('-inf')
    for epoch in range(args.epochs):
        if args.is_train == 1:
          model.train()
          epoch_train_loss = 0
          with tqdm(total=len(train_dataloader), desc="train") as t:
              for step, data in enumerate(train_dataloader):
                  optimizer.zero_grad()
                  img, html, text , label = data
                  label = label.to(args.device)
                  output = model(img, html, text)    # [batch_size,num_classes]
                  loss_train = lossfunc(output, label)  
                  loss_train.backward()
                  t.set_postfix(loss=loss_train.item())
                  t.update(1)
                  epoch_train_loss += loss_train.item()
                  optimizer.step()

                  if step % 100 == 0 and test_dataloader is not None:
                    logger.info("starting test: epoch:{}, step:{}".format(epoch, step))
                    with torch.no_grad():
                      model.eval()
                      total_test_loss = 0
                      for data in test_dataloader:
                        img, html, text , label = data
                        label = label.to(args.device)
                        output = model(img, html, text)
                        loss_test = lossfunc(output, label.long())
                        metric.add_data_piece(output, label.long())
                        total_test_loss += loss_test
                    
                    test_result = metric.get_metric()
                    dev_acc = test_result["acc"]
                    dev_f1 = test_result["f1"]
                    logger.info("end test, epoch:{}, step:{}, current_acc:{:.4f}, current_f1:{:.4f}".format(epoch, step, dev_acc, dev_f1))
                    if dev_f1 > dev_best_f1:
                      dev_best_f1 = dev_f1
                      print(f"current best acc-{dev_f1:.4f}, fs-{dev_f1:.4f}, epoch-{epoch}, step-{step}")
                      print("save model:", epoch, "./output/"+f"ckpt_{args.date}_{args.lr}_e{epoch}_s{step}_t{args.is_text}_i{args.is_img}_h{args.is_html}_{args.version}_acc_{dev_acc:.4f}_f1_{dev_best_f1:.4f}_{args.label_type}")
                      torch.save(model.state_dict(), "./output/"+f"ckpt_{args.date}_{args.lr}_e{epoch}_s{step}_t{args.is_text}_i{args.is_img}_h{args.is_html}_{args.version}_acc_{dev_acc:.4f}_f1_{dev_best_f1:.4f}_{args.label_type}")
                    metric.reset_epoch()
                        
          print("Epoch:{}, loss= {:.4f}".format(epoch, epoch_train_loss / len(train_dataloader)))
        
        if args.is_eval == 1:
          if test_dataloader is not None:
            with torch.no_grad():
              model.eval()
              total_test_loss = 0
              for data in tqdm(test_dataloader):
                img, html, text , label = data
                label = label.to(args.device)
                output = model(img, html, text)
                loss_test = lossfunc(output, label.long())
                metric.add_data_piece(output, label.long())
                # pred = torch.argmax(output,dim=1)
                # right += torch.sum(pred==data['label'].long())
                # all += len(data['label'])
                total_test_loss += loss_test
              # best_acc = max(best_acc,right/all)
            
            test_result = metric.get_metric()
            logger.info("Test set results:{} loss= {:.4f},acc:{:.4f},f1:{:.4f}".format(
                        str(epoch), total_test_loss.item() / len(test_dataloader), test_result["acc"], test_result["f1"]))
            
            dev_acc = test_result["acc"]
            dev_f1 = test_result["f1"]
          
            if dev_f1 > dev_best_f1 and args.is_train == 1:
                dev_best_f1 = dev_f1
                torch.save(model.state_dict(), "./output/"+f"ckpt_{args.lr}_epoch{epoch}_t{args.is_text}_i{args.is_img}_h{args.is_html}_{args.version}_acc_{dev_acc:.4f}_f1_{dev_best_f1:.4f}_{args.label_type}")
                print("save model:", epoch, "./output/"+f"ckpt_{args.lr}_epoch{epoch}_t{args.is_text}_i{args.is_img}_h{args.is_html}_{args.version}_acc_{dev_acc:.4f}_f1_{dev_best_f1:.4f}_{args.label_type}")
            
            metric.reset_epoch()
            
        scheduler.step()
        # logger.info("now acc:{}".format(str(right/all)))

    # best_result = metric.get_best_metric()
    logger.info(metric.get_best_metric_2_logger_format())
    metric.reset_all()


def config():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weight_decay", type=float, default=5e-4, help="the weight decay of training")
  parser.add_argument('--lrf', type=float, default=0.01, help="")
  parser.add_argument("--epochs", type=int, default=10, help="the epoch of training")
  parser.add_argument("--batch_size", type=int, default=100, help="the batch size of training")
  parser.add_argument("--html_node_text_embedding", action="store_true", default=False, help="use text embedding or not")
  parser.add_argument("--num_classes", type=int, default=3, help="the class num of task")

  parser.add_argument("--train_data_path", type=str, default="./datasets/train_58k_docid_score_dims", help="train_data_path")
  parser.add_argument("--test_data_path", type=str, default="./datasets/test_7k_docid_score_dims", help="test_data_path")

  parser.add_argument('--gat_nhid', type=int, default=8, help="")
  parser.add_argument('--gat_layers', type=int, default=2, help="")
  parser.add_argument('--gat_nfeat', type=int, default=8, help="")
  parser.add_argument('--gat_dropout', type=float, default=0.1, help="")
  parser.add_argument('--gat_nheads', type=int, default=8, help="")
  
  parser.add_argument('--bert_init_ckpt', type=str, default="/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain", help="")
  parser.add_argument('--bert_tokenize_path', type=str, default="/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain", help="")
  parser.add_argument('--vit_init_ckpt', type=str, default="/Vision-Transformer-pytorch/weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth", help="")
  
  parser.add_argument("--lr", type=float, default=0.001, help="the learning rate of training")
  parser.add_argument('--version', type=str, default="v3", help="lr1, lr2, lr3, lr4")
  parser.add_argument('--date', type=str, default="2024", help="") 
  parser.add_argument('--Uni_trained_ckpt', type=str, default="./output/ckpt_0.001_e4_s200_t1_i1_h1_v4_acc_0.6841_f1_0.6668_overScore", help="")
  parser.add_argument("--label_type", type=str, default="overScore", help="overScore, releScore, contScore, designScore, authScore")
  parser.add_argument('--is_train', type=int, default=0, help="")             # train=1,eval=0
  parser.add_argument('--Uni_para_reload', type=int, default=1, help="")      # train=0,eval=1
  
  parser.add_argument('--is_text', type=int, default=1, help="")
  parser.add_argument('--is_img', type=int, default=1, help="")
  parser.add_argument('--is_html', type=int, default=1, help="")
  
  parser.add_argument('--add_fc_layer', type=int, default=1, help="0, 1, 2")
  parser.add_argument('--is_eval', type=int, default=1, help="")
  parser.add_argument('--bert_trained_ckpt', type=str, default="./models/Bert/bert.ckpt.overScore_0.6601", help="bert init weight")
  parser.add_argument('--vit_trained_ckpt', type=str, default="./models/Vit/vit-4-0.5341.pth.overScore", help="vit init weight")                            # 最终版本
  parser.add_argument('--gat_trained_ckpt', type=str, default="./models/pyGAT/output/ckpt_l2_acc_0.5259_overScore", help="gat init weight")                                 # 最终版本
  parser.add_argument('--bert_para_reload', type=int, default=1, help="")
  parser.add_argument('--bert_para_freeze', type=int, default=1, help="")
  parser.add_argument('--vit_para_reload', type=int, default=1, help="")
  parser.add_argument('--vit_para_freeze', type=int, default=1, help="")
  parser.add_argument('--gat_para_reload', type=int, default=1, help="")
  parser.add_argument('--gat_para_freeze', type=int, default=1, help="")
  # GPU
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument('--device', type=str, default="cuda:0", help="")
  parser.add_argument('--gpu', type=str, default="0", help="")
  
  args = parser.parse_args()
  assert args.is_train  + args.Uni_para_reload == 1 
  return args

logger = loggerfunc()

def main():
  args = config()
  logger.info("args:{}".format(str(args)))
  logger.info("gpu:{}, device:{}".format(args.gpu, args.device))
  logger.info("bert_para_freeze:{}, bert_para_reload:{}".format(args.bert_para_freeze, args.bert_para_reload))
  logger.info("vit_para_freeze:{},  vit_para_reload:{}".format(args.vit_para_freeze, args.vit_para_reload))
  logger.info("gat_para_freeze:{},  gat_para_reload:{}".format(args.gat_para_freeze, args.gat_para_reload))
  
  logger.info("is train:{}, is eval:{}".format(args.is_train, args.is_eval))
  logger.info("model is text:{}, is_img:{}, is_html:{}".format(args.is_text, args.is_img, args.is_html))
  
  logger.info("train_data_path:  {}".format(args.train_data_path))
  logger.info("test_data_path:  {}".format(args.test_data_path))
  logger.info("label_type: {}".format(args.label_type))
  

  # 1.data process
  # 1.1 data path
  train_img_path, train_html_path, train_text_path, train_example_label  = read_data_path(args.train_data_path, config=args)
  test_img_path, test_html_path, test_text_path, test_example_label = read_data_path(args.test_data_path, config=args)

  img_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
  # 1.2 dataset
  train_dataset = MyDataset(img_paths=train_img_path,
                            html_paths=train_html_path,
                            text_paths=train_text_path,
                            example_labels=train_example_label,
                            img_transform=img_transform["train"],
                            config=args)
  test_dataset = MyDataset(img_paths=test_img_path,
                            html_paths=test_html_path,
                            text_paths=test_text_path,
                            example_labels=test_example_label,
                            img_transform=img_transform["test"],
                            config=args)
  # 1.3 dataloader
  nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
  logger.info('Using {} dataloader workers every process'.format(nw))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=False,
                                             collate_fn=train_dataset.collate_fn)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=False,
                                             collate_fn=test_dataset.collate_fn)
  
  # 2. model init and para reload
  model = UniQualityModel(
                text_encoder="bert",
                image_encoder="vit",
                html_encoder="gat",
                config=args).to(args.device)
  
  # 2.1 BERT
  bert_state_dict = torch.load(args.bert_trained_ckpt, map_location=torch.device("cpu"))
  if args.bert_para_reload == 1:
    model_dict_bert = model.state_dict()
    for bert_k, bert_v in bert_state_dict.items():
      if "bert_encoder."+bert_k in model_dict_bert:
        model_dict_bert["bert_encoder."+bert_k] = bert_v
    model_dict_bert["fc_v5.weight"] = bert_state_dict["fc.weight"]
    model_dict_bert["fc_v5.bias"] = bert_state_dict["fc.bias"]
    model.load_state_dict(model_dict_bert)
  else:
    print("bert_para_reload:", args.bert_para_reload)
  
  if args.bert_para_freeze == 1:
    for name, para in model.named_parameters():
        if "bert_encoder" in name or "fc_v5" in name:
            para.requires_grad_(False)
  else:
    print("bert_para_freeze:",args.bert_para_freeze)


  # 2.2 ViT
  model_dict_vit = model.state_dict()
  if args.vit_para_reload == 1:
    vit_state_dict = torch.load(args.vit_trained_ckpt, map_location=torch.device("cpu"))
    for vit_k, vit_v in vit_state_dict.items():
      if "image_encoder."+vit_k in model_dict_vit:
        model_dict_vit["image_encoder."+vit_k] = vit_v
    model_dict_vit["fc_v6.weight"] = vit_state_dict["head.weight"]
    model_dict_vit["fc_v6.bias"] = vit_state_dict["head.bias"]
  else:
    vit_state_dict = torch.load(args.vit_init_ckpt, map_location=torch.device("cpu"))
    print("vit_para_reload:", args.vit_para_reload)
    del_keys = ['head.weight', 'head.bias'] 
    for k in del_keys:
            del vit_state_dict[k]
    for vit_k, vit_v in vit_state_dict.items():
      if "image_encoder."+vit_k in model_dict_vit:
        model_dict_vit["image_encoder."+vit_k] = vit_v
  model.load_state_dict(model_dict_vit)

  if args.vit_para_freeze == 1:
    for name, para in model.named_parameters():
        if "image_encoder" in name or "fc_v6" in name:
            para.requires_grad_(False)
  else:
    print("vit_para_freeze:", args.vit_para_freeze)


  # 2.3 GAT
  if args.gat_para_reload == 1:
    gat_state_dict = torch.load(args.gat_trained_ckpt, map_location=torch.device("cpu"))
    model_dict_gat = model.state_dict()
    for gat_k, gat_v in gat_state_dict.items():
      if "html_encoder."+gat_k in model_dict_gat:
        model_dict_gat["html_encoder."+gat_k] = gat_v
    model_dict_gat["fc_v7.weight"] = gat_state_dict["cls.weight"]
    model_dict_gat["fc_v7.bias"] = gat_state_dict["cls.bias"]
    model.load_state_dict(model_dict_gat)
  else:
    print("gat_para_reload:", args.gat_para_reload)
  
  if args.gat_para_freeze == 1:
    for name, para in model.named_parameters():
        if "html_encoder" in name or "fc_v7" in name:
            para.requires_grad_(False)
  else:
    print("gat_para_freeze:", args.gat_para_freeze)
  
  # UniQuality
  if args.Uni_para_reload == 1:
    uni_state_dict = torch.load(args.Uni_trained_ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(uni_state_dict)
  else:
    print("uni_para_reload:", args.Uni_para_reload)

  train(args, train_loader, model, test_dataloader=test_loader)
if __name__ == "__main__":
  main()

import logging
from pathlib import Path
import datetime
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torchvision import transforms

from models.tokenization import BertTokenizer
import numpy as np
import json
import scipy.sparse as sp


def loggerfunc():
  def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file+'.txt')
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger
  if (not os.path.exists('./tmp')):
      os.mkdir('./tmp')
  if (not os.path.exists('./tmp/logs')):
      os.mkdir('./tmp/logs')
  log_file_name = str(datetime.datetime.now())[
      :-7].replace(' ', '_').replace(":", "_").replace("-", "_")
  logger = init_logger('./tmp/logs/' + log_file_name)
  return logger


def read_data_path(root_path: str, config=None):
  '''数据集train,test数据路径读取
    root_path: 数据路径
  '''
  dir = os.path.abspath(os.path.join(root_path, ".."))          # 获取上一级目录
  print("root_path:", root_path, "  dir:", dir)
  img_paths = []
  html_paths = []
  text_paths = []
  example_labels = []
  
  # screenshot name list
  screenshot_dir = os.path.join(dir,"screenshot")
  screenshot_list = [term.split(".png")[0] for term in os.listdir(screenshot_dir)]
  print("screenshot files:",len(screenshot_list))
  
  # text name list
  text_dir = os.path.join(dir,"text_new")
  text_list = [term.split(".txt")[0] for term in os.listdir(text_dir)]
  print("text files:",len(text_list))
  
  # html name list
  html_dir = os.path.join(dir,"html_feature")
  html_list = [term.split(".fea_adj")[0] for term in os.listdir(html_dir)]
  print("html files:",len(html_list))
  
  
  # 读取训练数据: docid_label
  with open(root_path,"r",encoding="utf-8") as f1:
    docid_label_data = f1.readlines()
    print("docid_label_data 数量:",len(docid_label_data))

  for line in tqdm(docid_label_data):
    line = line.strip().split("\t")
    if len(line) != 10:
      print(line)
    docid,url,query,title,overScore,inter,releScore,contScore,designScore,authScore = line
    assert config.label_type in ["overScore", "releScore", "contScore", "designScore", "authScore"]
    if config.label_type == "overScore":
      label = overScore
    elif config.label_type == "releScore":
      label = releScore
    elif config.label_type == "contScore":
      label = contScore
    elif config.label_type == "designScore":
      label = designScore
    elif config.label_type == "authScore":
      label = authScore

    assert docid in screenshot_list
    assert docid in text_list
    assert docid in html_list
    img_path = os.path.join(screenshot_dir, docid+".png")
    '''
    img = Image.open(img_path)
    try:
      if img.mode != 'RGB':
        img = img.convert('RGB')
    except:
      print("img_err:", img_path)
      continue
    '''
    html_path = os.path.join(html_dir, docid+".fea_adj")
    text_path = os.path.join(text_dir, docid+".txt")
    
    img_paths.append(img_path)
    html_paths.append(html_path)
    text_paths.append(text_path)
    example_labels.append(int(label))

  print("img examples:",len(img_paths))
  print("html examples:",len(html_paths))
  print("text examples:",len(text_paths))
  print("label examples:",len(example_labels))
  
  return img_paths, html_paths, text_paths, example_labels

# 数据进行transform变换
data_transform = {
  "train": transforms.Compose([transforms.Resize(256),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
  "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

class MyDataset(Dataset):
  """自定义数据集"""
  
  def __init__(self, img_paths: list, html_paths: list, text_paths: list, example_labels: list, img_transform=None, config=None):
    self.img_paths = img_paths
    self.html_paths = html_paths
    self.text_paths = text_paths
    self.example_labels = example_labels
    self.img_transform = img_transform
    self.config = config
    assert len(self.img_paths) == len(self.html_paths) == len(self.text_paths) == len(self.example_labels)

  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, index):
    img = self.img_process(self.img_paths[index])         # [C,H,W]
    html = self.html_process(self.html_paths[index])      # example:{"feature":[[n1],[n2]], "edges":[]}
    text = self.text_process(self.text_paths[index])      # (input_ids, segment_ids, input_mask)
    label = self.example_labels[index]
    
    return img, html, text, label
    
  def html_process(self, htm_file):
    # HTML处理: feature,edges,adj构建
    return get_html_dataset(htm_file)

  def text_process(self, text_file):
    # 文本处理
    return get_text_dataset(text_file, self.config.bert_tokenize_path)

  def img_process(self, img_file):
    # 图像处理
    img = Image.open(img_file)
    # # RGB为彩色图片，L为灰度图片
    if img.mode != 'RGB':
      img = img.convert('RGB')
    if self.img_transform is not None:
      img = self.img_transform(img)
    return img

  @staticmethod
  def collate_fn(batch):
    # 官方实现的default_collate可以参考
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    imgs, htmls, texts, labels = tuple(zip(*batch))
    '''
    print("imgs:",type(imgs), type(imgs[0]))
    print("htmls",type(htmls), type(htmls[0]))
    print("texts:",type(texts), type(texts[0]))
    print("labels:",type(labels), type(labels[0]))
    '''
    labels = torch.as_tensor(labels)    # [B]       需要确认多种实现
    
    # img处理
    imgs = torch.stack(imgs, dim=0)     # [B,C,H,W] [B,3,244,244]
    
    # text处理
    input_ids = torch.LongTensor([tem[0] for tem in texts])
    seg_ids = torch.LongTensor([tem[1] for tem in texts])
    mask_ids = torch.LongTensor([tem[2] for tem in texts])
    
    texts = (input_ids, seg_ids, mask_ids)
    
    # html处理
    return_htmls = {}
    node_length = []
    index = 0
    for data in htmls:                       # for html in batch_htmls:     [50,4]
        node_length.append(index)             
        index += data["feature"].shape[0]         # index:batch������html��node����
    return_htmls["feature"] = torch.cat([data["feature"] for data in htmls], dim=0)         #   ��batch������node����ƴ��
    all_adjs = torch.zeros(
        return_htmls["feature"].shape[0], return_htmls["feature"].shape[0])
    for index, node_index in enumerate(node_length):
        all_adjs[
            node_index: node_index + htmls[index]["adj"].shape[0],
            node_index: node_index + htmls[index]["adj"].shape[0],
        ] = htmls[index]["adj"]
    return_htmls["node_length"] = torch.tensor(node_length)
    return_htmls["adj"] = all_adjs

    return imgs, return_htmls, texts, labels              # imgs:[], htmls:{"feature":[], "edges":[], "adj":[]}, texts:[], labels:[]

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def get_text_dataset(path, bert_tokenize_path,  max_seq_len=512):
  with open(path,"r",encoding="utf-8") as f1:
    data = f1.readlines()
    assert len(data) == 1
    line = data[0].strip().split("\t")
    assert len(line) == 2
    title, content = line
    tokenizer = BertTokenizer.from_pretrained(bert_tokenize_path)
    title_token = tokenizer.tokenize(title)
    content_token = tokenizer.tokenize(content)
    _truncate_seq_pair(title_token, content_token, max_seq_len-3)
    
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    
    for token in title_token:
      tokens.append(token)
      segment_ids.append(0)
      
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    for token in content_token:
      tokens.append(token)
      segment_ids.append(1)
    
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_len:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    res = (input_ids, segment_ids, input_mask)
    return res

def get_html_dataset(file_path):
    tensor_column_name = ["feature", "edges"]
    reserved_key = ["feature", "edges", "adj"]
    now_piece_data = {}
    strs = open(file_path,"r",encoding="utf-8").readlines()[0]
    raw_docid, raw_example_idx, raw_feature, raw_edges = strs.strip().split("\t")         # raw_feature,raw_edges--->feature,adj
    now_piece_data["feature"] = json.loads(raw_feature)
    now_piece_data["edges"] = json.loads(raw_edges)       # todo
    for key in tensor_column_name:
        if key in now_piece_data:
            now_piece_data[key] = torch.tensor(now_piece_data[key], dtype=torch.float32)
            if key == "feature":
                idx = np.array(now_piece_data[key][:, 0], dtype=np.int32)
                idx_map = {j: i for i, j in enumerate(idx)}
                now_piece_data[key] = now_piece_data[key][:, 1:]
    feature = now_piece_data["feature"]
    edges_unordered = np.array(json.loads(raw_edges))
    try: 
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32,).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(feature.shape[0], feature.shape[0]),dtype=np.float32,)
        adj = (adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
    except:
        print("error:",raw_docid)
        return
    now_piece_data["adj"] = adj
    reserved_data = {}
    for key in reserved_key:
        reserved_data[key] = now_piece_data[key]
    # print("now_piece_data:",now_piece_data)
    num_features = reserved_data["feature"].shape[1]        # 1008
    style_features = 240                                    # 240
    return reserved_data

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


if __name__ == "__main__":
  device = "cuda:0"
  train_path = "/search/odin/wyg/webdata/ACL_final_dataset_labe_spider/test_7k_docid_score_dims"
  img_paths, html_paths, text_paths, example_labels = read_data_path(train_path)
  dataset = MyDataset(img_paths=img_paths,
                      html_paths=html_paths,
                      text_paths=text_paths,
                      example_labels=example_labels,
                      img_transform=data_transform["train"])
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=MyDataset.collate_fn)
  for i, example in enumerate(dataloader):
    if i < 2 :
      print(example[0].shape)
      print(example[1]["feature"].shape, example[1]["adj"].shape)
      print(example[2][0].shape)      # example[2]:  batch_text,  example[2][0]:batch_text_第一个example
      print(example[3])
    else:
      break
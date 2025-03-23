#/usr/env/bin python
# coding:utf-8

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models.ViT import vit_base_patch16_224_in21k as ViTModel
from models.GAT import WebGAT
from models.BERT import BertModel
from models.tokenization import BertTokenizer


class BERT_Model(nn.Module):
    def __init__(self, config):
        super(BERT_Model, self).__init__()
        self.device = config.device
        self.bert = BertModel.from_pretrained(config.bert_init_ckpt)
        print("init bert para form: {}".format(config.bert_init_ckpt))
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0].to(self.device)               # input_ids
        seg = x[1].to(self.device)                   # segment_ids
        mask = x[2].to(self.device)                  # mask_ids
        _, pooled = self.bert(context, token_type_ids = seg, attention_mask=mask, output_all_encoded_layers=False)
        # out = self.fc1(pooled)
        out = pooled
        return out


class UniQualityModel(nn.Module):
  def __init__(self,
               text_encoder=None,             # BERT相关配置or路径
               image_encoder=None,            # ViT相关配置or路径
               html_encoder=None,             # GAT相关配置or路径
               config=None                    # 其他配置
              ):
    super(UniQualityModel, self).__init__()
    self.config = config
    self.device = config.device
    self.bert_encoder = BERT_Model(self.config)                                       # bert
    self.image_encoder = ViTModel(num_classes=config.num_classes, has_logits=False)     # vit
    self.html_encoder = WebGAT(nhid=config.gat_nhid,                                  # gat: return: node_0_embedding: [B,1,nhid_*_nheads]
                               nclass=config.num_classes,
                               nfeat=config.gat_nfeat,
                               dropout=config.gat_dropout,
                               nheads=config.gat_nheads,
                               alpha=0.2,
                              style_feat=240,
                              config=self.config)
    # 融合输出分类
    # v1: bert_768 + vit_768 + gat_100 = 1663; 再mlp:1663*num_classes
    # v2: attention[bert,vit,gat]
    # v3: bert,vit,gat三者都通过一层mlp映射到同一个维度大小,再通过权重系数相加,接mlp

    self.fc_v1 = nn.Linear(768+768+64, 3)          # text+img+html
    
    self.fc_v2 = nn.Linear(768+768, 3)          # text+img
    self.fc_v3 = nn.Linear(768+64, 3)          # text+html
    self.fc_v4 = nn.Linear(768+64, 3)          # img+html
    
    self.fc_v5 = nn.Linear(768,3)              # only_bert
    self.fc_v6 = nn.Linear(768,3)              # only_vit
    self.fc_v7 = nn.Linear(64,3)              # only_gat
    
    # add 1 fc layer
    self.add_1_fc1 = nn.Linear(1600, 128)
    self.add_1_fc2 = nn.Linear(128, 3)
    
    # add 2 fc layers
    self.add_2_fc1 = nn.Linear(1600, 768)
    self.add_2_fc2 = nn.Linear(768, 64)
    self.add_2_fc3 = nn.Linear(64, 3)
    
    
    self.flag_text = config.is_text 
    self.flag_img = config.is_img
    self.flag_html = config.is_html
    self.add_fc_layer = config.add_fc_layer
    print("self.flag_text:{}, self.flag_img:{}, self.flag_html:{}, self.add_fc_layer:{}".format(self.flag_text, self.flag_img, self.flag_html, self.add_fc_layer))
  
  def forward(self, img, html, text, config=None):
    text_cls = self.bert_encoder(text)                                                                                                          # [B, 768]
    image_cls = self.image_encoder(img.to(self.device))                                                                                         # [B,768]
    html_cls = self.html_encoder(html["feature"].to(self.device), html["adj"].to(self.device), html["node_length"].to(self.device))             # [B,64]
    
    # v1: text + html + img
    if self.flag_text == 1 and self.flag_img == 1 and self.flag_html == 1:
      text_image_html_emb = torch.cat([text_cls, image_cls, html_cls], dim=-1)
      if self.add_fc_layer == 1:
        out = self.add_1_fc1(text_image_html_emb)
        out = self.add_1_fc2(out)
      elif self.add_fc_layer == 2:
        out = self.add_2_fc1(text_image_html_emb)
        out = self.add_2_fc2(out)
        out = self.add_2_fc3(out)
      else:
        out = self.fc_v1(text_image_html_emb)
      
    # v2: text + img
    elif self.flag_text == 1 and self.flag_img == 1 and self.flag_html == 0:
      text_img_emb = torch.cat([text_cls, image_cls], dim=-1)
      out = self.fc_v2(text_img_emb)
      
    # v3: text + html
    elif self.flag_text == 1 and self.flag_img == 0 and self.flag_html == 1:
      text_html_emb = torch.cat([text_cls, html_cls], dim=-1)
      out = self.fc_v3(text_html_emb)
    
    # v4: img + html
    elif self.flag_text == 0 and self.flag_img == 1 and self.flag_html == 1:
      img_html_emb = torch.cat([image_cls, html_cls], dim=-1)
      out = self.fc_v4(img_html_emb)
    
    
    # v5: text
    elif self.flag_text == 1 and self.flag_img == 0 and self.flag_html == 0:           # only_bert
      out = self.fc_v5(text_cls)
    
    # v6: vit
    elif self.flag_text == 0 and self.flag_img == 1 and self.flag_html == 0:          # only_img
      out = self.fc_v6(image_cls)

    # v7: gat
    elif self.flag_text == 0 and self.flag_img == 0 and self.flag_html == 1:          # only_gat
      out = self.fc_v7(html_cls)
    
    return out




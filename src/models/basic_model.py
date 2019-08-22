# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/26
# @Author  : Jiaqi&Zecheng
# @File    : basic_model.py
# @Software: PyCharm
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pytorch_transformers import XLNetModel, XLNetTokenizer

from src.rule import semQL as define_rule


class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        self.lm_model = XLNetModel.from_pretrained('xlnet-base-cased')
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.CLS = self.tokenizer.encode('<cls>')[0]
        self.PAD = self.tokenizer.encode('<pad>')[0]
        self.SEP = self.tokenizer.encode('<sep>')[0]
        self.emb_cache = {}
        

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ), 0)

        return embedding_differ

    def encode(self, src_sents_var, src_sents_len, q_onehot_project=None):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """
        src_token_embed = self.gen_x_batch(src_sents_var)

        if q_onehot_project is not None:
            src_token_embed = torch.cat([src_token_embed, q_onehot_project], dim=-1)

        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        # src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], -1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings, (last_state, last_cell)

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))

        return padding_result

    def gen_x_batch(self, src_sents, column_names=None, table_names=None):
        B = len(src_sents)
        if column_names and table_names:
            for sent, columns, tables in zip(src_sents, column_names, table_names):
                key = str(sent + columns + tables)
                if key in self.emb_cache:
                    continue
                ids = [self.CLS]
                seg_ids = []
                for seg in sent:
                    seg_id = self.tokenizer.encode(' '.join(seg))
                    seg_ids.append(seg_id)
                    ids += seg_id
                
                col_ids = []
                for column in columns:
                    col_id = self.tokenizer.encode(' '.join(column))
                    col_ids.append(col_id)
                    ids += [self.SEP] + col_id
                
                tab_ids = []
                for table in tables:
                    tab_id = self.tokenizer.encode(' '.join(table))
                    tab_ids.append(tab_id)
                    ids += [self.SEP] + tab_id
                ids.append(self.SEP)

                with torch.no_grad():
                    ids_tensor = torch.tensor([ids])
                    if self.args.cuda:
                        ids_tensor = ids_tensor.cuda()
                    embs = self.lm_model(ids_tensor)[0][0]
                embs = embs.cpu().numpy()[1:]  # remove CLS
                
                sent_embs = []
                idx = 0
                for seg_id in seg_ids:
                    l = len(seg_id)
                    sent_embs.append(embs[idx: idx + l].mean(0))
                    idx += l
                
                sent_embs = np.stack(sent_embs)
                assert len(sent_embs) == len(sent)

                col_embs = []
                for col_id in col_ids:
                    l = len(col_id)
                    col_embs.append(embs[idx + 1: idx + l + 1].mean(0))
                    idx += l + 1
                
                col_embs = np.stack(col_embs)
                assert len(col_embs) == len(columns)

                tab_embs = []
                for tab_id in tab_ids:
                    l = len(tab_id)
                    tab_embs.append(embs[idx + 1: idx + l + 1].mean(0))
                    idx += l + 1
                
                tab_embs = np.stack(tab_embs)
                assert len(tab_embs) == len(tables)

                assert len(embs) == idx + 1

                self.emb_cache[key] = (sent_embs, col_embs, tab_embs)
                
        else:
            for sent in src_sents:
                key = str(sent)
                if key in self.emb_cache:
                    continue
                ids = [self.CLS]
                seg_ids = []
                for seg in sent:
                    seg_id = self.tokenizer.encode(' '.join(seg))
                    seg_ids.append(seg_id)
                    ids += seg_id
                with torch.no_grad():
                    ids_tensor = torch.tensor([ids])
                    if self.args.cuda:
                        ids_tensor = ids_tensor.cuda()
                    embs = self.lm_model(ids_tensor)[0][0]
                embs = embs.cpu().numpy()[1:]  # remove CLS
                sent_embs = []
                idx = 0
                for seg_id in seg_ids:
                    l = len(seg_id)
                    sent_embs.append(embs[idx: idx + l].mean(0))
                    idx += l
                
                sent_embs = np.stack(sent_embs)
                assert len(sent_embs) == len(sent)
                self.emb_cache[key] = sent_embs

        def pad_embs(embs):
            max_len = max([len(emb) for emb in embs])
            val_emb_array = np.zeros((B, max_len, self.args.col_embed_size), dtype=np.float32)
            for i in range(B):
                for t in range(len(embs[i])):
                    val_emb_array[i, t, :] = embs[i][t]

            val_inp = torch.from_numpy(val_emb_array)
            if self.args.cuda:
                val_inp = val_inp.cuda()
            return val_inp

        if column_names and table_names:
            sent_embs_, col_embs_, tab_embs_ = [], [], []
            for sent, columns, tables in zip(src_sents, column_names, table_names):
                key = str(sent + columns + tables)
                sent_embs, col_embs, tab_embs = self.emb_cache[key]
                sent_embs_.append(sent_embs)
                col_embs_.append(col_embs)
                tab_embs_.append(tab_embs)

            return pad_embs(sent_embs_), pad_embs(col_embs_), pad_embs(tab_embs_)
        else:
            sent_embs_ = []
            for sent in src_sents:
                key = str(sent)
                sent_embs = self.emb_cache[key]
                sent_embs_.append(sent_embs)
            return pad_embs(sent_embs_)


        # max_len = max(val_len)

        # val_emb_array = np.zeros((B, max_len, self.args.col_embed_size), dtype=np.float32)
        # for i in range(B):
        #     for t in range(len(val_embs[i])):
        #         val_emb_array[i, t, :] = val_embs[i][t]
        # val_inp = torch.from_numpy(val_emb_array)
        # if self.args.cuda:
        #     val_inp = val_inp.cuda()
        # return val_inp

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

import os
import re
import math
import data
import time
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransactionSignatures(pl.LightningModule):

    def __init__(self, hparams):
        super(TransactionSignatures, self).__init__()

        self.hparams = hparams
        print(vars(hparams))

        # Variable output sizes are determined by the data and set in prepare_data()
        self.feature_set = {'merchant_name':
                                {'enabled': True,
                                'output_size': None,
                                'loss_weight': hparams.merchant_name_loss_weight},
                            'user_reference':
                                {'enabled': hparams.include_user_context,
                                'output_size': None,
                                'loss_weight': 1},
                            'eighth_of_day':
                                {'enabled': hparams.include_eighth_of_day,
                                'output_size': 9,
                                'loss_weight': hparams.tod_loss_weight},
                            'day_of_week':
                                {'enabled': hparams.include_day_of_week,
                                'output_size': 8,
                                'loss_weight': hparams.dow_loss_weight},
                            'amount':
                                {'enabled': hparams.include_amount,
                                'output_size': 1,
                                'loss_weight': hparams.amount_loss_weight},
                            'mcc':
                                {'enabled':hparams.include_mcc,
                                'output_size': None,
                                'loss_weight': hparams.mcc_loss_weight},
                            'proj_2D':
                                {'enabled':hparams.include_proj_2D,
                                'output_size': 2,
                                'loss_weight': hparams.proj_2D_loss_weight}
                            }

        # Provide path to directory containing data files if loading locally
        # See data.py for more detail
        self.dataset = data.LoadDataset(self.hparams.sample_size,
                                        self.hparams.dataset_name,
                                        self.hparams.seq_len,
                                        local_source=None)
        self.dataset.download_dict()

        self.feature_set['merchant_name']['output_size'] = self.dataset.nmerchant
        self.feature_set['user_reference']['output_size'] = self.dataset.nusers
        self.feature_set['mcc']['output_size'] = self.dataset.nmcc

        if hparams.use_pretrained_embeddings is False:
            self.merchant_embedding = nn.Embedding(self.dataset.nmerchant, hparams.embedding_size)
        else:
            embeddings = torch.tensor(self.dataset.dictionary['merchant_embeddings']).float()
            self.merchant_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.aux_feat_size = 0
        if self.feature_set['eighth_of_day']['enabled']:
            self.aux_feat_size += 2
        if self.feature_set['day_of_week']['enabled']:
            self.aux_feat_size += 2
        if self.feature_set['amount']['enabled']:
            self.aux_feat_size += 1

        if self.aux_feat_size > 0:
            self.aux_embedding = nn.Linear(self.aux_feat_size, self.hparams.embedding_size)

        if self.feature_set['user_reference']:
            self.user_embedding = nn.Embedding(self.dataset.nusers, self.hparams.embedding_size)

        if self.feature_set['mcc']:
            self.mcc_embedding = nn.Embedding(self.dataset.nmcc, self.hparams.embedding_size)

        self.src_mask = None
        self.input_dropout = nn.Dropout(hparams.input_dropout)
        self.layer_dropout = nn.Dropout(hparams.layer_dropout)
        encoder_layers = TransformerEncoderLayer(hparams.embedding_size,
                                                 hparams.nhead,
                                                 hparams.nhid,
                                                 hparams.layer_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, hparams.nlayers)

        # Initialize postitional encoder
        # 5000 = max_seq_len
        pe = torch.zeros(5000, hparams.embedding_size)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hparams.embedding_size, 2).float()
                * (-math.log(10000.0) / hparams.embedding_size)
            )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.decoders = nn.ModuleDict({})
        for feature, config in self.feature_set.items():
            # excluding user_reference because input and target are the same
            if config['enabled'] and feature != 'user_reference':
                decoder_layers = []
                decoder_name = feature + '_decoder'
                for i, x in enumerate(range(self.hparams.ndecoder_layers-1)):
                    decoder_layers.append(nn.Linear(self.hparams.embedding_size,
                                             self.hparams.embedding_size))
                    decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(self.hparams.embedding_size,
                                         config['output_size']))
                setattr(self, decoder_name, nn.Sequential(*decoder_layers))
                self.decoders[feature] = getattr(self, decoder_name)

                if feature == 'proj_2D':
                    projection_decoder = []
                    projection_decoder.append(nn.Linear(self.hparams.embedding_size,
                                                        int(self.hparams.embedding_size/2)))
                    projection_decoder.append(nn.ReLU())
                    projection_decoder.append(nn.Linear(int(self.hparams.embedding_size/2), 2))
                    setattr(self, 'proj_2D_decoder', nn.Sequential(*projection_decoder))
                    self.decoders[feature] = getattr(self, 'proj_2D_decoder')


    def positional_encoder(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.layer_dropout(x)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self,
                inputs,
                has_src_mask=True,
                src_key_padding_mask=None):

        mask_sum = src_key_padding_mask.sum(dim=1)
        #print('masked elems: {}'.format(mask_sum))
        #print(mask_sum.max())
        assert mask_sum.max() != self.hparams.seq_len
        #print(mask_sum.min())
        assert src_key_padding_mask.sum(dim=1).eq(self.hparams.seq_len).byte().any() == False

        if has_src_mask:
            device = inputs['merchant_name'].device
            if self.src_mask is None or self.src_mask.size(0) != inputs['merchant_name'].size(1):
                mask = self._generate_square_subsequent_mask(inputs['merchant_name'].size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.merchant_embedding(inputs['merchant_name']) * math.sqrt(self.hparams.embedding_size)
        merchant_embedding = src.clone()

        if self.feature_set['user_reference']['enabled']:
            user_src = self.user_embedding(inputs['user_reference']) * math.sqrt(self.hparams.embedding_size)
            # swap batch and elem dims (0, 1) to be expandable elementwise)
            src = src.permute(1,0,2)
            src = src + user_src.expand_as(src)
            # swap back
            src = src.permute(1,0,2)

        if self.feature_set['mcc']['enabled']:
            mcc_src = self.mcc_embedding(inputs['mcc']) * math.sqrt(self.hparams.embedding_size)
            src = src + mcc_src

        auxilliary_features = []

        if self.feature_set['eighth_of_day']['enabled']:
            auxilliary_features.append(inputs['eighth_of_day'])
        if self.feature_set['day_of_week']['enabled']:
            auxilliary_features.append(inputs['day_of_week'])
        if self.feature_set['amount']['enabled']:
            auxilliary_features.append(inputs['amount'])

        if self.aux_feat_size > 0:
            aux_inputs = torch.cat(auxilliary_features, 2)
            aux_src = self.aux_embedding(aux_inputs) * math.sqrt(self.hparams.embedding_size)
            src = src + aux_src

        src = self.positional_encoder(src)
        #print('scr: {}'.format(src.size()))
        #print('key_padding: {}'.format(src_key_padding_mask.size()))
        src = src.permute(1,0,2)
        #print('attn_mask: {}'.format(self.src_mask.size()))
        transformer_output = self.transformer_encoder(src,
                                                      self.src_mask,
                                                      src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.permute(1,0,2)
        #print('transformer_output: {}'.format(transformer_output.size()))

        outputs = {}
        trace_output = tuple()
        for feature, decoder in self.decoders.items():
            key_name = feature + '_output'
            if feature == 'proj_2D':
                decoder_output = decoder(merchant_embedding)
            else:
                decoder_output = decoder(transformer_output)
            #print('decoder_output {0}: {1}'.format(feature, decoder_output.size()))
            outputs[feature] = decoder_output
            trace_output += (decoder_output,)

        return outputs


    def cross_entropy_loss(self, outputs, targets, masks):
        masks = 1.0 - masks.float()
        loss = F.cross_entropy(outputs, targets, reduce='none')
        loss = loss * masks
        masks_sum = masks.sum(dim=1)
        loss = loss.sum(dim=1) / masks_sum
        print('cross_ent loss: {}'.format(loss.mean()))
        return loss.mean()


    def mse_loss(self, outputs, targets, masks):
        #print('mse output: {}'.format(output.size()))
        #print('mse target: {}'.format(targets.size()))
        #print('mse masks: {}'.format(masks.size()))
        masks = 1.0 - masks.float()
        loss = F.mse_loss(outputs, targets, reduce='none')
        loss = loss * masks
        masks_sum = masks.sum(dim=1)
        loss = loss.sum(dim=1) / masks_sum
        print('mse loss: {}'.format(loss.mean()))
        return loss.mean()

    def proj_loss(self, outputs, masks):
        # minimize the distinace between points in the projected space

        N = torch.tensor(outputs.size(0)).float() # batch size
        d = outputs.size(1) # projection dim
        S = outputs.size(2) # sequence length

        intraseq_dists = []
        for idx, seq in enumerate(outputs):
            mask = ~masks[idx] # size
            S_nonpad = mask.sum()
            #print(S_nonpad)
            if S_nonpad < 256:
                seq = seq[:,:S_nonpad]
            #print(seq)
            x = seq.permute(1,0) #size: [S_nonpad,d]
            #print(x.size())
            x = x.unsqueeze(0).expand(S_nonpad,S_nonpad,d)
            y = x.permute(1,0,2)
            dist = torch.pow(x - y, 2).sum(2) # size: [S_nonpad, S_nonpad]
            #print(dist)
            #print(dist.sum())
            avg_dist = dist.sum() / (S_nonpad**2 - (S_nonpad-1))
            #print(avg_dist)
            intraseq_dists.append(torch.sqrt(avg_dist))

        intraseq_dists = torch.stack(intraseq_dists, dim=0).type_as(outputs)
        #print(intraseq_dists)
        #print(intraseq_dists.mean())
        return intraseq_dists.mean()


    def recall_at_k(self, outputs, k, targets, masks):
        batch_recall = torch.tensor(0).type_as(outputs)
        # loop over sequnces in the batch and tokens in the sequence
        for i, (target_seq, mask) in enumerate(zip(targets, masks)): # target_seq size: [S]
            target_seq = target_seq.repeat(k,1) # target_seq size: [k,S]
            output_seq = outputs[i] # output_seq size: [S,C]
            _, indices = torch.topk(output_seq, k) # indices size: [S,k]
            permute_indices = indices.permute(1,0) # size: [k,S]
            eq_target_seq = target_seq.eq(permute_indices) # size: [k,S]
            eq_target_seq = eq_target_seq * mask.view(1,-1) # size: [k,S]
            correct_count = torch.sum(eq_target_seq, dim=0).float() # size: [S]
            seq_recall = torch.div(correct_count.sum(), mask.sum()) # size: scalar
            batch_recall += seq_recall

        batch_recall = torch.div(batch_recall.float(), targets.size(0))
        return batch_recall


    def training_step(self, train_batch, batch_idx):
        inputs, targets, padding_masks = train_batch
        #print('padding_masks: {}'.format(padding_masks))
        outputs = self.forward(inputs,
                               src_key_padding_mask=padding_masks,
                               has_src_mask=True)
        padding_masks = padding_masks#.view(-1)
        logs = {}
        general_loss = torch.tensor(0.).type_as(outputs['merchant_name'])
        for feature, logits in outputs.items():
            key = '{0}_train_loss'.format(feature)
            logits_view = logits.permute(0,2,1)

            if feature=='proj_2D':
                loss = self.proj_loss(logits_view, padding_masks)
            else:
                targets_view = targets[feature]#.view(-1)
                if feature=='amount':
                    loss = self.mse_loss(torch.squeeze(logits_view), torch.squeeze(targets_view), padding_masks)
                else:
                    loss = self.cross_entropy_loss(logits_view, targets_view, padding_masks)

            logs[key] = loss
            loss_weight = torch.tensor(self.feature_set[feature]['loss_weight']).type_as(loss)
            weighted_loss = loss * loss_weight
            if torch.isnan(weighted_loss).any():
                print('\n\nLogits: {0}\n\n'.format(logits.size()))
                print('\n\nTartget: {0}\n\n'.format(targets[feature].size()))
                print('{0} returned NaN loss'.format(feature))
                #weighted_loss = torch.tensor(self.hparams.epsilon).type_as(loss)
            general_loss += weighted_loss
        logs['train_loss'] = general_loss

        masked_targets = (1-padding_masks.float())
        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'], masked_targets)
            key = 'merchant_name_train_recall_at_{0}'.format(str(k.item()))
            logs[key] = recall

        return {'loss': general_loss, 'log': logs}


    def training_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        return {'loss': logs['train_loss_epoch'], 'log': logs}


    def validation_step(self, val_batch, batch_idx):
        inputs, targets, padding_masks = val_batch
        outputs = self.forward(inputs,
                               src_key_padding_mask=padding_masks,
                               has_src_mask=True)

        logs = {}
        general_loss = torch.tensor(0.).type_as(outputs['merchant_name'])
        for feature, logits in outputs.items():
            key = '{0}_val_loss'.format(feature)
            logits_view = logits.permute(0,2,1)

            if feature=='proj_2D':
                loss = self.proj_loss(logits_view, padding_masks)
            else:
                targets_view = targets[feature]#.view(-1)
                if feature=='amount':
                    loss = self.mse_loss(torch.squeeze(logits_view), torch.squeeze(targets_view), padding_masks)
                else:
                    loss = self.cross_entropy_loss(logits_view, targets_view, padding_masks)

            logs[key] = loss
            loss_weight = torch.tensor(self.feature_set[feature]['loss_weight']).type_as(logits)
            weighted_loss = loss * loss_weight
            if torch.isnan(weighted_loss).any():
                print('\n\nLogits: {0}\n\n'.format(logits))
                print('\n\nTartget: {0}\n\n'.format(targets[feature]))
                print('{0} returned NaN loss'.format(feature))
                #weighted_loss = torch.tensor(self.hparams.epsilon).type_as(loss)
            general_loss += weighted_loss
        logs['val_loss'] = general_loss

        masked_targets = (1-padding_masks.float())
        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'], masked_targets)
            key = 'merchant_name_val_recall_at_{0}'.format(str(k.item()))
            logs[key] = recall

        #self.predict(self.val_data)

        return {'val_loss': general_loss, 'log': logs}


    def validation_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        return {'val_loss':logs['val_loss_epoch'], 'log': logs}


    def test_step(self, test_batch, batch_idx):
        inputs, targets, padding_masks = test_batch
        outputs = self.forward(inputs,
                               src_key_padding_mask=padding_masks,
                               has_src_mask=True)

        logs = {}
        general_loss = torch.tensor(0.).type_as(outputs['merchant_name'])
        for feature, logits in outputs.items():
            key = '{0}_test_loss'.format(feature)
            logits_view = logits.permute(0,2,1)

            if feature=='proj_2D':
                loss = self.proj_loss(logits_view, padding_masks)
            else:
                targets_view = targets[feature]#.view(-1)
                if feature=='amount':
                    loss = self.mse_loss(torch.squeeze(logits_view), torch.squeeze(targets_view), padding_masks)
                else:
                    loss = self.cross_entropy_loss(logits_view, targets_view, padding_masks)

            logs[key] = loss
            loss_weight = torch.tensor(self.feature_set[feature]['loss_weight']).type_as(logits)
            weighted_loss = loss * loss_weight
            if torch.isnan(weighted_loss).any():
                print('\n\nLogits: {0}\n\n'.format(logits))
                print('\n\nTartget: {0}\n\n'.format(targets[feature]))
                print('{0} returned NaN loss'.format(feature))
                weighted_loss = torch.tensor(self.hparams.epsilon).type_as(loss)
            general_loss += weighted_loss
        logs['test_loss'] = general_loss

        masked_targets = (1-padding_masks.float())
        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'], masked_targets)
            key = 'merchant_name_test_recall_at_{0}'.format(str(k.item()))
            logs[key] = recall

        return {'test_loss': general_loss, 'log': logs}


    def test_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        return {'log': logs}


    def predict(self, predict_data):
        prediction_inputs = DataLoader(predict_data,
                          batch_size=1,
                          num_workers=4)
        prediction_outputs = []
        for sample_idx, sample in enumerate(prediction_inputs):
            input, target, padding_mask = sample
            logits = self.forward(input,
                                   src_key_padding_mask=padding_mask,
                                   has_src_mask=True)

            _, indices = torch.topk(logits['merchant_name'], 1)
            merchant_names = []
            for idx in indices.squeeze():
                merchant_names.append(self.dataset.dictionary['idx2merchant'][idx.item()])
            prediction_outputs.append(merchant_names)

        print(prediction_outputs[0])

        return prediction_outputs


    def prepare_data(self):
        self.dataset.download_data()
        self.train_data = self.dataset.train
        self.val_data = self.dataset.val
        #print(self.val_data[0])
        self.test_data = self.dataset.test

        print('Enabled features: {}'.format(self.feature_set))
        pass


    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=True,
                          shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=True)


    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4,
                          drop_last=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      eps=self.hparams.epsilon)
        return optimizer

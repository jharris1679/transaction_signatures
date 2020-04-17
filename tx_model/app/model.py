import os
import math
import data
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
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
                                'loss_weight': 1},
                            'user_reference':
                                {'enabled': hparams.include_user_context,
                                'output_size': None,
                                'loss_weight': 1},
                            'eighth_of_day':
                                {'enabled': hparams.include_eighth_of_day,
                                'output_size': 9,
                                'loss_weight': 1},
                            'day_of_week':
                                {'enabled': hparams.include_day_of_week,
                                'output_size': 8,
                                'loss_weight': 1},
                            'amount':
                                {'enabled': hparams.include_amount,
                                'output_size': 1,
                                'loss_weight': 1},
                            'sys_category':
                                {'enabled': hparams.include_sys_category,
                                'output_size': None,
                                'loss_weight': 1}
                            }

        self.features = data.Features(hparams.data,
                                      hparams.seq_len,
                                      self.feature_set,
                                      hparams.data_cache,
                                      hparams.isLocal)

        self.feature_set['merchant_name']['output_size'] = self.features.ntoken
        self.feature_set['user_reference']['output_size'] = self.features.nusers
        self.feature_set['sys_category']['output_size'] = self.features.ncat

        if self.hparams.use_pretrained_embeddings is False:
            self.token_embedding = nn.Embedding(self.features.ntoken, hparams.embedding_size)
        else:
            embeddings = torch.tensor(self.features.dictionary.token_embeddings).float()
            self.token_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

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
            self.user_embedding = nn.Embedding(self.features.nusers, self.hparams.embedding_size)

        if self.feature_set['sys_category']:
            self.cat_embedding = nn.Embedding(self.features.ncat, self.hparams.embedding_size)

        self.src_mask = None
        self.input_dropout = nn.Dropout(hparams.input_dropout)
        self.layer_dropout = nn.Dropout(hparams.layer_dropout)
        encoder_layers = TransformerEncoderLayer(hparams.embedding_size,
                                                 hparams.nhead,
                                                 hparams.nhid,
                                                 hparams.layer_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, hparams.nlayers)

        # Initialize postition encoder
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

        self.decoders = {}
        for feature, config in self.feature_set.items():
            # excluding user_reference because input and target are the same
            if config['enabled'] and feature != 'user_reference':
                decoder = []
                for i, x in enumerate(range(self.hparams.ndecoder_layers-1)):
                    decoder.append(nn.Linear(self.hparams.embedding_size,
                                             self.hparams.embedding_size))
                    decoder.append(nn.ReLU())
                decoder.append(nn.Linear(self.hparams.embedding_size,
                                         config['output_size']))
                self.decoders[feature] = nn.Sequential(*decoder)


    def positional_encoder(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.layer_dropout(x)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def encode_cyclical(self, X):
        X = X.type(torch.FloatTensor)
        sin = torch.sin(X)
        cos = torch.cos(X)
        return torch.squeeze(torch.stack((sin, cos),2))


    def amount_scaler(self, X):
        # Assumes min = 0 and max = 3000
        Xsc = X / 3000
        return torch.unsqueeze(Xsc, dim=-1)


    def forward(self, inputs, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.token_embedding(inputs['merchant_name']) * math.sqrt(self.hparams.embedding_size)
        print(src.device)

        if self.feature_set['user_reference']['enabled']:
            user_src = self.user_embedding(inputs['user_reference']) * math.sqrt(self.hparams.embedding_size)
            # swap batch and elem dims (0, 1) to be expandable elementwise
            src = src.permute(1,0,2)
            src = src + user_src.expand_as(src)
            # swap back
            src = src.permute(1,0,2)

        if self.feature_set['sys_category']['enabled']:
            cat_src = self.cat_embedding(inputs['sys_category']) * math.sqrt(self.hparams.embedding_size)
            src = src + cat_src

        auxilliary_features = []
        if self.feature_set['eighth_of_day']['enabled']:
            eighth_of_day = self.encode_cyclical(inputs['eighth_of_day'])
            auxilliary_features.append(eighth_of_day)
        if self.feature_set['day_of_week']['enabled']:
            day_of_week = self.encode_cyclical(inputs['day_of_week'])
            auxilliary_features.append(day_of_week)
        if self.feature_set['amount']['enabled']:
            amount_scaled = self.amount_scaler(inputs['amount'])
            auxilliary_features.append(amount_scaled)

        if self.aux_feat_size > 0:
            auxilliary_features = torch.squeeze(torch.cat(auxilliary_features, 2))
            aux_src = self.aux_embedding(auxilliary_features) * math.sqrt(self.hparams.embedding_size)
            src = src + aux_src

        src = self.positional_encoder(src)
        transformer_output = self.transformer_encoder(src, self.src_mask)

        decoder_outputs = {}
        for feature, decoder in self.decoders.items():
            output = decoder(transformer_output[:,0])
            decoder_outputs[feature] = F.log_softmax(output, dim=-1)

        return decoder_outputs


    def cross_entropy_loss(self, output, targets):
        criterion = nn.CrossEntropyLoss()
        return criterion(output, targets)


    def mse_loss(self, output, targets):
        criterion = nn.MSELoss()
        return criterion(output, targets)


    def recall_at_k(self, outputs, k, targets):
        correct_count = 0
        # loop over batches
        for i, x in enumerate(targets):
            values, indices = torch.topk(outputs[i], k)
            if x in indices:
                correct_count += 1
        return correct_count / self.hparams.batch_size


    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self.forward(inputs)

        logs = {}
        general_loss = torch.tensor(0.)
        for feature, logits in outputs.items():
            key = feature + '_train_loss'
            if feature=='amount':
                amount_targets = self.amount_scaler(targets[feature])
                loss = self.mse_loss(logits, amount_targets)
            else:
                loss = self.cross_entropy_loss(logits, targets[feature])
            logs[key] = loss
            loss *= self.feature_set[feature]['loss_weight']
            general_loss += loss
        general_loss /= len(outputs)
        logs['train_loss'] = general_loss

        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'])
            key = 'merchant_name_train_recall_at_{0}'.format(str(k.item()))
            logs[key] = torch.tensor(recall)

        return {'loss': general_loss, 'log': logs}


    def training_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        return {'log': logs}


    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self.forward(inputs)

        logs = {}
        general_loss = torch.tensor(0.)
        for feature, logits in outputs.items():
            key = '{0}_val_loss'.format(feature)
            if feature=='amount':
                amount_targets = self.amount_scaler(targets[feature])
                loss = self.mse_loss(logits, amount_targets)
            else:
                loss = self.cross_entropy_loss(logits, targets[feature])
            logs[key] = loss
            loss *= self.feature_set[feature]['loss_weight']
            general_loss += loss
        general_loss /= len(outputs)
        logs['val_loss'] = general_loss

        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'])
            key = 'merchant_name_val_recall_at_{0}'.format(str(k.item()))
            logs[key] = torch.tensor(recall)

        return {'val_loss': general_loss, 'log': logs}


    def validation_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        self.sync_logs()
        return {'log': logs}


    def test_step(self, test_batch, batch_idx):
        inputs, targets = test_batch
        outputs = self.forward(inputs)

        logs = {}
        general_loss = 0.
        for feature, logits in outputs.items():
            key = '{0}_test_loss'.format(feature)
            if feature=='amount':
                amount_targets = self.amount_scaler(targets[feature])
                loss = self.mse_loss(logits, amount_targets)
            else:
                loss = self.cross_entropy_loss(logits, targets[feature])
            logs[key] = loss.item()
            loss *= self.feature_set[feature]['loss_weight']
            general_loss += loss
        general_loss /= len(outputs)
        logs['test_loss'] = general_loss

        for k in self.hparams.kvalues:
            recall = self.recall_at_k(outputs['merchant_name'], k, targets['merchant_name'])
            key = 'merchant_name_test_recall_at_{0}'.format(str(k.item()))
            logs[key] = torch.tensor(recall)

        return {'log': logs}


    def test_epoch_end(self, outputs):
        logs = {}
        for metric in outputs[0]['log'].keys():
            avg_metric = torch.stack([x['log'][metric] for x in outputs]).mean()
            key = '{0}_epoch'.format(metric)
            logs[key] = avg_metric

        self.sync_logs()
        return {'log': logs}


    def prepare_data(self):
        self.train_data = self.features.train_data
        self.val_data = self.features.val_data
        print(self.val_data[0])
        self.test_data = self.features.test_data

        print('Enabled features: {}'.format(self.feature_set))
        pass


    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4)


    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4)


    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=4)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def sync_logs(self):
        # unhappy hack to get logs into GCS
        path = os.path.join('lightning_logs/', self.logger.name, self.logger.version)
        print('Syncing {} to GCS'.format(path))
        cmd_string = 'gsutil -m -q cp -r {0} gs://tensorboard_logging/lightning_logs/{1}/'
        copy_command = cmd_string.format(path, self.logger.name)
        subprocess.run(copy_command.split())
        pass

import os
import torch
import pytorch_lightning as pl
import model
from argparse import ArgumentParser
from tensorboardX import SummaryWriter


def main(args, experiment_name):
    logger = pl.loggers.TensorBoardLogger('lightning_logs', name='tx_model', version=experiment_id)
    model = model.TransactionSignatures(hparams=args)
    trainer = pl.Trainer(logger=logger)
    trainer.fit(model)


if __name__ == '__main__':
    experiment_id = os.getenv('EXP_ID')

    parser = ArgumentParser()

    # adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    # parametrize the network
    parser.add_argument('--experiment_id', type=str, default=experiment_id,
                        help='Name of input data')
    parser.add_argument('--data', type=str, default='merchant_seqs_by_tx',
                        help='Name of input data')
    parser.add_argument('--data_cache', action='store_true',
                        help='Use cached BQ table if option is present')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='size of word embeddings')
    parser.add_argument('--use_pretrained_embeddings', action='store_true',
                        help='Use pretrained embeddings or not')
    parser.add_argument('--include_user_context', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_eighth_of_day', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_day_of_week', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_amount', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_sys_category', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='number of transformer layers')
    parser.add_argument('--ndecoder_layers', type=int, default=1,
                        help='number decoder of layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='sequence length')
    parser.add_argument('--input_dropout', type=float, default=0,
                        help='dropout applied to input sequences (0 = no dropout)')
    parser.add_argument('--layer_dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--kvalues', type=int, default=torch.tensor([1,3,10]),
                        help='Values of k for calculating recall at k')
    parser.add_argument('--isLocal', action='store_true',
                        help='Run on a small subset of the data')

    args = parser.parse_args()

    # train
    main(args, experiment_id)

import os
import torch
import pytorch_lightning as pl
import model
import gcs
from argparse import ArgumentParser
from tensorboardX import SummaryWriter



def main(args):
    logs_path = os.path.join('lightning_logs', args.model_name,  args.experiment_id)
    logger = pl.loggers.TensorBoardLogger('lightning_logs',
                                          name=args.model_name,
                                          version=args.experiment_id)

    checkpoint_filename = args.experiment_id + '-{epoch}-{val_loss:.2f}'
    checkpoints_path = os.path.join('checkpoints', args.model_name, checkpoint_filename)
    ckpt_callback = pl.callbacks.ModelCheckpoint(filepath=checkpoints_path)
    gcs_callback = gcs.LogSyncCallback(args.log_interval,
                                       args.model_name,
                                       args.experiment_id)

    if args.isLocal==True:
        gpus = 0
        precision = 32
    else:
        gpus = args.num_gpus
        precision = 16

    initialized_model = model.TransactionSignatures(hparams=args)
    #initialized_model.prepare_data()
    #logger.experiment.add_graph(initialized_model, next(iter(initialized_model.val_dataloader())))
    trainer = pl.Trainer(logger=logger,
                        checkpoint_callback=ckpt_callback,
                        callbacks=[gcs_callback],
                        max_epochs=args.epochs,
                        gpus=gpus,
                        distributed_backend='dp',
                        precision = precision)
    trainer.fit(initialized_model)


if __name__ == '__main__':
    experiment_id = os.getenv('EXP_ID')

    parser = ArgumentParser()

    # adds all the trainer options as default arguments (like max_epochs)
    parser = pl.Trainer.add_argparse_args(parser)

    # parametrize the network
    parser.add_argument('--experiment_id', type=str, default=experiment_id,
                        help='Name of current run')
    parser.add_argument('--model_name', type=str, default='tx_model',
                        help='Name of model')
    parser.add_argument('--data_cache', action='store_true',
                        help='Use cached BQ table if option is present')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--architecture', type=str, default='Transformer',
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
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
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
                        help='No calls to GCP')

    # --------- These are tracked in the experiment log ----------- #
    # ------------------------------------------------------------- #
    parser.add_argument('--data', type=str, default='merchant_seqs_by_tx_power',
                        help='Name of input data')
    parser.add_argument('--input_dropout', type=float, default=0,
                        help='dropout applied to input sequences (0 = no dropout)')
    parser.add_argument('--layer_dropout', type=float, default=0.02,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='sequence length')
    parser.add_argument('--merchant_name_loss_weight', type=int, default=1,
                        help='Turn on feature')
    parser.add_argument('--tod_loss_weight', type=int, default=0.1,
                        help='Turn on feature')
    parser.add_argument('--dow_loss_weight', type=int, default=0.1,
                        help='Turn on feature')
    parser.add_argument('--amount_loss_weight', type=int, default=0.01,
                        help='Turn on feature')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='number of transformer layers')
    parser.add_argument('--ndecoder_layers', type=int, default=1,
                        help='number decoder of layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--epsilon', type=int, default=1e-6,
                        help='For numerical stability')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='How much of the data to download. Files must already exist in this amount')

    args = parser.parse_args()

    # train
    main(args)

from model import TransactionSignatures
from argparse import ArgumentParser
import pytorch_lightning as pl

parser = ArgumentParser()

# adds all the trainer options as default arguments (like max_epochs)
parser = pl.Trainer.add_argparse_args(parser)

parser.add_argument('--sample_size', type=int, default=100,
                    help='How much of the data to download. Files must already exist in this amount')

args = parser.parse_args()

ckpt_path = '/home/checkpoints/saved/drop_slot-epoch=0-val_loss=4.54.ckpt'
model = TransactionSignatures.load_from_checkpoint(ckpt_path, hparams=args)
print('model loaded')

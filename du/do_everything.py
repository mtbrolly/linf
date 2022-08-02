
import argparse
from build_model import build_model
from prep_data import prep_data
from train_model import train_model
# from results import results
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='current', type=str,
                    help='desired location of output file')
parser.add_argument('--checkpoint_file', default=None, type=str,
                    help='checkpoint from which to continue training')
args = parser.parse_args()

model_name = args.model_name
model_dir = "du/models/" + model_name + "/"
fig_dir = model_dir + "figures/"

# Create relevant directories if they do not exist.
if not Path(model_dir).exists():
    Path(model_dir).mkdir(parents=True)
if not Path(fig_dir).exists():
    Path(fig_dir).mkdir(parents=True)


checkpoint_file = args.checkpoint_file

if not checkpoint_file:
    build_model(model_dir)
    prep_data(model_dir)

train_model(model_dir, checkpoint_file)
# results(model_dir, checkpoint_file)

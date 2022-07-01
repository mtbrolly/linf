import argparse
from build_model import build_model
from prep_data import prep_data
from train_model import train_model
from results import results
from pathlib import Path

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='current', type=str,
                    help='desired location of output file')
args = parser.parse_args()

model_name = args.model_name
model_dir = "velocity_differences/models/" + model_name + "/"
fig_dir = model_dir + "figures/"

# Create relevant directories if they do not exist.
if not Path(model_dir).exists():
    Path(model_dir).mkdir(parents=True)
if not Path(fig_dir).exists():
    Path(fig_dir).mkdir(parents=True)


results(model_dir)

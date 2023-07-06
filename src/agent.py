import torch
import os
import nle.dataset as nld
from nle.nethack import tty_render

dbfile = 'nle.db'
dataset_name = 'tester_dataset'
path = '/home/pavel/dev/isp/nle_data/nle_data'

if not nld.db.exists(dbfile):
    nld.db.create(dbfile)
    nld.add_nledata_directory(path=path, name=dataset_name, filename=dbfile)
db_conn = nld.db.connect(filename=dbfile)
dataset = nld.TtyrecDataset(
    dataset_name,
    batch_size=32,
    seq_length=32,
    dbfilename=dbfile,
)
minibatch = next(iter(dataset))
batch_idx = 0
time_idx = 0
chars = minibatch['tty_chars'][batch_idx, time_idx]
colors = minibatch['tty_colors'][batch_idx, time_idx]
cursor = minibatch['tty_cursor'][batch_idx, time_idx]

print(tty_render(chars, colors, cursor))
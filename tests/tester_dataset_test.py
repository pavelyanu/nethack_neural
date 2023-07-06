import os
import unittest
import nle.dataset as nld
from nle.nethack import tty_render

class TesterDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dbfile = 'test_nle.db'
        self.dataset_name = 'tester_dataset'
        self.path = '/data/nld-aa-taster/nle_data'
        if not nld.db.exists(self.dbfile):
            nld.db.create(self.dbfile)
            nld.add_nledata_directory(path=self.path, name=self.dataset_name, filename=self.dbfile)
        else:
            raise FileExistsError(f'Test database file {self.dbfile} already exists. Please remove it first.')
        
    def test_ttyrec_dataset(self):
        db_conn = nld.db.connect(filename=self.dbfile)
        dataset = nld.TtyrecDataset(
            self.dataset_name,
            batch_size=32,
            seq_length=32,
            dbfilename=self.dbfile,
        )
        minibatch = next(iter(dataset))

    def tearDown(self) -> None:
        os.remove(self.dbfile)

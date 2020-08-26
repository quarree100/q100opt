
import os

from q100opt.cli import main
from q100opt.setup_model import load_csv_data

basedir = os.path.dirname(__file__)


def test_main():
    assert main([]) == 0


def test_import_csv():
    dir_import = os.path.join(basedir, '_files/import_csv')
    dic = load_csv_data(dir_import)
    assert dic['Source']['label'][0] == 'heat'

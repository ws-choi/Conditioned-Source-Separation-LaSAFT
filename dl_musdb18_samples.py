import musdb
from lasaft.utils.functions import mkdir_if_not_exists
mkdir_if_not_exists('etc')
musdb.DB(root='etc/musdb18_dev', download=True)

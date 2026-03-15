""" to setup databases required for tfit scores
"""

from tfitpy.datasets.setup import install
from tfitpy.datasets.pair_cache import build,load

import os

if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")
    # install(data_path)
    build(data_path,n_jobs=-1, batch_size=25000)
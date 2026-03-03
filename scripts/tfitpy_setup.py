""" to setup databases required for tfit scores
"""

from tfitpy.datasets.setup import install
import os

if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")
    install(data_path)
from multiprocessing.context import assert_spawning
import pickle
import os
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

# filename = "privacy/configurations/configurations.pkl"
EXPIRY_DATE = datetime.strptime("01/01/2050", "%d/%m/%Y")

if datetime.now() <= EXPIRY_DATE:
    print("This is DEMO version of PRISM (Synthetic Data Generation and Privacy Tool")
    print("Copyright ZS: All the files assocaited with PRISM are use for DEMO purposes. No part of the code/notebooks are to be circulated, quoted or reproduced for distribution without prior written approval from ZS")
else:
    raise ImportError("DEMO Licence of PRISM has expired. Please reach out to ZS Admin")
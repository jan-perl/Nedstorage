# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#interactieve vorm nedsumm01
# -

import pandas as pd
import numpy as np
import re
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import subprocess
import requests
import json
import os

from ipywidgets import Dropdown, Button, Output, interact
from IPython.display import display

# +
#by using run, all intermediate results will be shown
# #%run './nedsumm01.ipynb'
# -

#using import, 
import nedsumm01


# +
# # !pip install pandas matplotlib ipywidgets
# -

def recalc(my2_inst_opw,my2_inst_long,my2_inst_short):
    nedsumm01.run_again (
       nedsumm01.landyrframe.copy(),
       my2_inst_opw,
       my2_inst_long,
       my2_inst_short     )
interact(
   recalc ,
       my2_inst_opw=Dropdown(options=nedsumm01.param_opw_df.index.values.tolist(), description='Opwek mix:'),
       my2_inst_long=Dropdown(options=nedsumm01.param_longdf.index.values.tolist(), description='Long-term:'),
       my2_inst_short=Dropdown(options=nedsumm01.param_shortdf.index.values.tolist(), description='Short-term:')                   
)



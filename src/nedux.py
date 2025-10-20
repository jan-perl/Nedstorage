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

from ipywidgets import Dropdown, Button, Output
from IPython.display import display

# +
#by using run, all intermediate results will be shown
# #%run './nedsumm01.ipynb'
# -

#using import, 
import nedsumm01

# !pip install pandas matplotlib ipywidgets

#droplist_l = Dropdown(options=list(nedsumm01.param_longdf['inst'].unique()), description='Long-term:')
droplist_l = Dropdown(options=nedsumm01.param_longdf.index.values.tolist(), description='Long-term:')
refresh = Button(description="Refresh", button_style='success')
output_interact = Output(layout={'border': '1px solid black'}) 


@output_interact.capture()
def on_refresh_champ(x):
    print (x)
    output_interact.clear_output()
    with output_champ:
        long = droplist_l.value
        v2=nedsumm01.run_again (nedsumm01.landyrframe.copy(),long,'A')  
#        widgets.interactive_output(v2)


display(droplist_l, refresh, output_interact)
refresh.on_click(on_refresh_champ)



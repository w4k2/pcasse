#!/bin/bash
python -W ignore plot.py
cd figures
montage 100-0.png 100-1.png 100-2.png 150-0.png 150-1.png 150-2.png 200-0.png 200-1.png 200-2.png -geometry +2+2  megaplot.png

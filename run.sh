#!/bin/bash

i=$1
python tools_dyall_initial_smallCAS.py ${i}
python fci_smallcas.py
python tools_dyall_initial_dmat.py ${i}
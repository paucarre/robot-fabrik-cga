#!/bin/bash
sudo apt install graphviz
pip install poetry==1.5.1
export PATH=$PATH:$HOME/.local/bin/
poetry --directory=dev-gpu update
poetry --directory=dev-gpu install
poetry --directory=dev-gpu shell
#!/bin/bash
pip install poetry==1.5.1
export PATH=$PATH:$HOME/.local/bin/
poetry update
poetry install
poetry shell
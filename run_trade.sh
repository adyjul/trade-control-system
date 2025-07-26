#!/bin/bash
cd /root/trade-control-system
source venv/bin/activate
python strategy/predictor.py
python strategy/executor.py

#!/bin/bash
cd /root/trade-control-system
source venv/bin/activate
# python3 -m strategy.run_trade
python3 -m strategy.predictor
python3 -m strategy.executor

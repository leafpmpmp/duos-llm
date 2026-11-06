#!/bin/bash
# Stop a running compile_mm.sh in WSL.
pkill -f compile_mm.sh
pkill -f model_transform.py
pkill -f run_calibration.py
pkill -f model_deploy.py
pkill -f model_runner.py
pkill -f tpuc-opt
sleep 2
pkill -9 -f model_transform.py
pkill -9 -f model_deploy.py
pkill -9 -f run_calibration.py
pkill -9 -f tpuc-opt
exit 0

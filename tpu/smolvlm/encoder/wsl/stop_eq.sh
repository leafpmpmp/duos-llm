pkill -f seq_eq.sh
pkill -f compile_eq.sh
pkill -f run_calibration
pkill -f model_transform
pkill -f model_deploy
pkill -f model_runner
pkill -f tpuc-opt
sleep 2
pkill -9 -f run_calibration
pkill -9 -f model_transform
pkill -9 -f model_deploy
pkill -9 -f tpuc-opt
echo "left: $(pgrep -fc 'seq_eq|compile_eq|run_calibration|model_transform|model_deploy|model_runner|tpuc-opt')"

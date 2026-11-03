# Block 5 only: BF16 GELU and BF16 fc2, with the original KL thresholds.
# Usage: bash fix_l05.sh LAYER
L=$1
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
cd ~/duosight-tpu/work/e512/"$L" || exit 1
cp cali_table.kl cali_table
python3 /mnt/d/duosight-eval/tpu/qtable_eq.py gelu > qt_c.txt
echo "/fc2/Conv_output_0_Conv BF16" >> qt_c.txt
model_deploy.py --mlir block.mlir --quantize INT8 --calibration_table cali_table --quantize_table qt_c.txt \
	--processor cv181x --test_input test_in.npz --test_reference top_outputs.npz --tolerance 0.0,-1.0 \
	--model block_eq.cvimodel > cc.log 2>&1 || { echo "$L deploy FAILED"; tail -3 cc.log | cut -c1-140; exit 1; }
model_runner.py --input test_in.npz --model block_eq.cvimodel --output host_eq.npz > r.log 2>&1
python3 /mnt/d/duosight-eval/tpu/check_eq.py "$L"
cp block_eq.cvimodel /mnt/d/duosight-eval/tpu/out/e512/"$L".cvimodel
ls -la block_eq.cvimodel | awk '{print $5}'

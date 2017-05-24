PRHW_DIR="${HOME}/PRHW_im2txt/data/prhw/tokenized"
MODEL_DIR="${HOME}/PRHW_im2txt/model/tokenized_baseline"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
python ~/PRHW_im2txt/evaluate.py \
  --input_file_pattern="${PRHW_DIR}/val-?????-of-00001" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval"


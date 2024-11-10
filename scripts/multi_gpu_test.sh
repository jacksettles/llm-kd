export MASTER_ADDR=$(hostname -i)             # Use localhost for single-node setup
export MASTER_PORT=29501                      # Pick an available port

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

python gpt2_train.py --ff_dim 768 --num_heads 4 --num_blocks 12 --embed_dim 768 --num_epochs 5 --distill 1 --teacher_model saved_models/rnng/tokenized_rnng.pt --savepath ptb/throwaway-test-2gpu-distilled --alpha 0.4 --break_value 5 --trainfile data/tokenized_data/ptb-train.pkl --valfile data/tokenized_data/ptb-val.pkl
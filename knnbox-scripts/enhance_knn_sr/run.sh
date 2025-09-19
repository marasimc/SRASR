# this line will speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..    
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/symbolicregression/end2end_epoch-30000
OPTOR_MAX_K=4
CONST_MAX_K=64
BS=64
COMBINER_LOAD_DIR="$PROJECT_PATH/save-models/combiner/enhanced-adaptive/end2end_max-k-${OPTOR_MAX_K}-${CONST_MAX_K}_bs${BS}_learnt_3w-data"

CHECKPOINT_PREFIX=checkpoint_last-
STEPS=500
BEAM_SIZE=16

# run strogatz benchmark
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common_sr/run.py --eval_on_pmlb True \
                   --pmlb_data_type strogatz \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True \
                   --knn_type enhance \
                   --knn_mode inference \
                   --knn_datastore_path $DATASTORE_SAVE_PATH \
                   --knn_max_k $MAX_K \
                   --knn_combiner_path $COMBINER_LOAD_DIR \
                   --optor_max_k $OPTOR_MAX_K \
                   --const_max_k $CONST_MAX_K \
                   --checkpoint_prefix $CHECKPOINT_PREFIX \
                   --bs $BS \
                   --beam_size $BEAM_SIZE \
                   --steps $STEPS


# run standard benchmark
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common_sr/run.py --eval_on_standard True \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True \
                   --knn_type enhance \
                   --knn_mode inference \
                   --knn_datastore_path $DATASTORE_SAVE_PATH \
                   --knn_max_k $MAX_K \
                   --knn_combiner_path $COMBINER_LOAD_DIR \
                   --optor_max_k $OPTOR_MAX_K \
                   --const_max_k $CONST_MAX_K \
                   --checkpoint_prefix $CHECKPOINT_PREFIX \
                   --bs $BS \
                   --beam_size $BEAM_SIZE \
                   --steps $STEPS

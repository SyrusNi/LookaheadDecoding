srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 -t 4:00:00 /bin/bash

python minimal.py #no Lookahead decoding
USE_LADE=1 LOAD_LADE=1 python minimal.py #use Lookahead decoding, 1.6x speedup
USE_LADE=1 LOAD_LADE=1 python -m ipdb minimal.py

python draft.py
USE_LADE=1 LOAD_LADE=1 python draft.py
USE_LADE=1 LOAD_LADE=1 python -m ipdb draft.py

#download data 
wget https://raw.githubusercontent.com/lm-sys/FastChat/v0.2.31/fastchat/llm_judge/data/mt_bench/question.jsonl -O mtbench.jsonl 

# eval
export CUDA=0
export LADE=1
export LEVEL=5
export WIN=15
export GUESS=15
export FLASH=0
export PP=0
CUDA_VISIBLE_DEVICES=$CUDA USE_LADE=$LADE python eval_mtbench.py \
    --model-path models/vicuna-7b-v1.3 --model-id \
    vicuna-7b-v1.3-level-$LEVEL-win-$WIN-guess-$GUESS-f$FLASH-pp$CUDA \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-pp $PP

# eval baseline
python applications/eval_mtbench.py \
    --model-path models/vicuna-7b-v1.3 \
    --model-id baseline-vicuna-7b-v1.3 \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-pp $PP
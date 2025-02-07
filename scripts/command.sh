srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 -t 4:00:00 /bin/bash

python minimal.py #no Lookahead decoding
USE_LADE=1 LOAD_LADE=1 python minimal.py #use Lookahead decoding, 1.6x speedup

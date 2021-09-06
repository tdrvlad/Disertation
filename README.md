docker run \
--gpus device=1 \
-it \
-v ~/Desktop/knosis-workspace/Disertation/:/Disertation \
-p8888:8888 \
--name knosis-class-gpu01 \
--rm \
knosis-classification-interactive
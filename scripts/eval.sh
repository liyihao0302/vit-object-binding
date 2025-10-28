#!/usr/bin/env bash

# Make sure this script is executable: 
# chmod +x train.sh

# List of layer indices to iterate over
LAYER_LIST=(0 6 12 18 23)

for L in "${LAYER_LIST[@]}"
do
  echo "Running evaluation with model.num_layer[0] = ${L}"

  # Example command:
  #  - 'main.py' is your Python entry point (it has @hydra.main etc.)
  #  - We override 'model.num_layer' from config.yaml 
  #  - The first element is set to L, and the rest remain the same 
  #    as in your original config definition, for example:
  #      model:
  #        num_layer: [11, 2, 3, 4, ..., 23]
  #
  # If your config has 23 elements after index 0, be sure to keep them consistent:
  
  python ../src/main.py \
    model.num_layer="[${L}, 2]" device=cuda:0 trainer.device=cuda:0 probe.mode=quadratic mode=eval_all
  
  # If you need to pass more Hydra overrides or command-line arguments,
  # just keep chaining them after python main.py, e.g.:
  #   python main.py \
  #       model.num_layer="[${L}, 2, ...]" \
  #       hydra.run.dir="./results/${L}"
  
  echo "Done evaluation with layer index: ${L}"
  echo "======================================"
done

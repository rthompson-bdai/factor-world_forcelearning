#!/bin/bash
#  Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

envs=(  #"Assembly" \
     #"Basketball" \
    # "CoffeePush" \
    # "BoxClose" \
    # "StickPull" \
    # "PegInsertSide" \
    # "Soccer" \
    "button-press" \
    #"pick-place" \
    #"bin-picking" \
    # "button-press-topdown" \
    # "button-press-topdown-wall" \
    #"door-lock" \
    #"door-open" \
    # "door-unlock" \
    #"drawer-close" \
    # "drawer-open" \
    #"faucet-close" \
    #"faucet-open" \
    #"handle-press" \
    #"handle-pull" \
    # "handle-pull-side" \
    #"lever-pull" \
    # "window-close" \
    #"window-open" \
    )

factors=(
    "None"
    # "arm_pos" \
    # "camera_pos" \
    # "distractor_pos"  \
    # "floor_texture" \
    # #"object_texture" \
    # "table_pos" \
    # "table_texture" \
    # "light" \
    # "object_size"
)


for env in ${envs[@]}; do
    for factor in ${factors[@]}; do
    (
        gsutil cp -r \
        /workspaces/bdai/projects/foundation_models/src/force_learning/ibrl_forcelearning/mw_main/bc_data/metaworld/${env}_${factor}_frame_stack_1_96x96_end_on_success/* \
        gs://bdai-common-storage/visuomotor/datasets/factorworld/${env}_${factor}
    )
    done
done
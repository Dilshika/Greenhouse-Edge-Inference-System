
#!/bin/bash

set -e

for outer in {1..6}; do
    for inner in {1..3}; do

        OUTER=$(printf "%02d" $outer)
        INNER=$(printf "%02d" $inner)

        TRAIN="data/first_edition/rotating_splits/outer_${OUTER}/inner_${INNER}/fit.csv"
        VALIDATION="data/first_edition/rotating_splits/outer_${OUTER}/inner_${INNER}/validation.csv"
        TEST="data/first_edition/rotating_splits/outer_${OUTER}/test_seen.csv"

        OUTPUT="results/first_edition/outer_${OUTER}/inner_${INNER}"

        echo "======================================"
        echo "Running Outer ${OUTER} - Inner ${INNER}"
        echo "======================================"

        # TinyML
        python train_tinyml.py \
            --train "$TRAIN" \
            --validation "$VALIDATION" \
            --test "$TEST" \
            --output-dir "$OUTPUT/tinyml" \
            --lookback 12 \
            --horizon 1 \
            --interval-minutes 5

        # XGBoost
        python train_xgboost.py \
            --train "$TRAIN" \
            --validation "$VALIDATION" \
            --test "$TEST" \
            --output-dir "$OUTPUT/xgboost" \
            --lookback 12 \
            --horizon 1 \
            --interval-minutes 5

        # LSTM
        python train_recurrent.py \
            --train "$TRAIN" \
            --validation "$VALIDATION" \
            --test "$TEST" \
            --output-dir "$OUTPUT/lstm" \
            --lookback 12 \
            --horizon 1 \
            --interval-minutes 5 \
            --model=lstm

        # GRU
        python train_recurrent.py \
            --train "$TRAIN" \
            --validation "$VALIDATION" \
            --test "$TEST" \
            --output-dir "$OUTPUT/gru" \
            --lookback 12 \
            --horizon 1 \
            --interval-minutes 5 \
            --model=gru

    done
done

echo "All training runs completed!"
:q
:xq


#!/bin/bash
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the list of optimizers in optimizer directory
OPTIMIZERS=()
for FILE in ${BASE_DIR}/optimizer/*.yml
do
    OPTIMIZERS+=($(basename ${FILE}))
done

FORMULAE="$(python ${BASE_DIR}/formula.py)"
for OPTIMIZER in "${OPTIMIZERS[@]}"
do
    for FORMULA in ${FORMULAE}
    do
        echo "${OPTIMIZER} - ${FORMULA}"
        ${BASE_DIR}/test_optimizer_compatibility.sh --optimizer ${OPTIMIZER} --formula ${FORMULA}
    done
done

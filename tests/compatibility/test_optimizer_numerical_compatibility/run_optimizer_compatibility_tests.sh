#!/bin/bash
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the list of optimizers in optimizer directory
OPTIMIZERS=()
for FILE in $(dirname ${BASE_DIR})/optimizer/*.yml
do
    OPTIMIZERS+=(${FILE})
done

FORMULAE="$(python ${BASE_DIR}/formula.py)"
for FORMULA in ${FORMULAE}
do
    for OPTIMIZER in "${OPTIMIZERS[@]}"
    do
        echo "${FORMULA} - ${OPTIMIZER}"
        ${BASE_DIR}/test_optimizer_compatibility.sh --optimizer ${OPTIMIZER} --formula ${FORMULA}
    done
done

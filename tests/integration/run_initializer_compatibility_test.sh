#!/bin/bash
set -u

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${BASE_DIR}/data/initializer"
TEST_DIR="${BASE_DIR}/test_initializer_compatibility"
TEST_COMMAND="${TEST_DIR}/test_initializer_compatibility.sh"

RETURN=0
for FILE in ${DATA_DIR}/*.yml
do
    "${TEST_COMMAND}" "${FILE}"
    if [[ ! $? = 0 ]]; then
	RETURN=1
    fi
done

exit ${RETURN}

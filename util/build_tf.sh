#!/bin/bash

set -e
set -x

if [[ $(uname) == Darwin ]]; then {
    bash util/build_tf_mac.sh
} else {
    bash util/build_tf_u1604.sh
} fi

#!/bin/bash

# builds tensorflow
# assumes you already configured it

# handles building/installing coriander, prior to building the main tensorflow package

# doesnt create teh python wheel. You'll need to run util/build_wheel.sh to build that

set -e
set -x

if [[ $(uname) == Darwin ]]; then {
    bash util/build_tf_mac.sh
} else {
    bash util/build_tf_u1604.sh
} fi

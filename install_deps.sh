#!/bin/bash

# installs dependencies
# tested on Mac Sierra and Ubuntu 16.04

set -e
set -x

if [[ $(uname) == Darwin ]]; then {
    bash util/install_deps_mac.sh
} else {
    bash util/install_deps_u1604.sh
} fi

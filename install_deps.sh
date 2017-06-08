#!/bin/bash

set -e
set -x

if [[ $(uname) == Darwin ]]; then {
    bash util/install_deps_mac.sh
} else {
    bash util/install_deps_u1604.sh
} fi

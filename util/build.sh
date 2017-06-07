#!/bin/bash

if [[ $(uname) == Darwin ]]; then {
    bash util/build_mac.sh
} else {
    bash util/build_1604.sh
} fi

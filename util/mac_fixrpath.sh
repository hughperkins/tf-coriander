#!/bin/bash

TF_PYPKG_DIR=$1

set -x
set -e

# patch on _pywrap_tensorflow
for libname in cocl clew easycl clblast; do {
    old_rpath=$(otool -L ${TF_PYPKG_DIR}/python/_pywrap_tensorflow.so | grep ${libname} | awk '{print $1}')
    echo old_rpath ${old_rpath}
    install_name_tool -change \
        ${old_rpath} \
        "@loader_path/../third_party/coriander/lib${libname}.dylib" \
        ${TF_PYPKG_DIR}/python/_pywrap_tensorflow.so
} done
otool -L ${TF_PYPKG_DIR}/python/_pywrap_tensorflow.so

function patch_lib {
    # libname, eg: cocl
    # (ie, excludes .dylib suffix, and lib prefix)
    targetlibname=$1
    relpath=$2
    libs=$3
    echo libname $libname
    echo relpath $relpath
    echo libs $libs

    target_dylib=${TF_PYPKG_DIR}/third_party/coriander/lib${targetlibname}.dylib
    for libname in ${libs}; do {
        otool -L ${target_dylib}
        old_rpath=$(otool -L ${target_dylib} | grep ${libname} | awk '{print $1}')
        echo old_rpath ${old_rpath}
        new_path="@loader_path/${relpath}/lib${libname}.dylib"
        echo new path $new_path
        if [[ x${old_rpath} == x ]]; then {
            install_name_tool -add_rpath \
                "${new_path}" \
                ${target_dylib}
        } else {
            install_name_tool -change \
                ${old_rpath} \
                "${new_path}" \
                ${target_dylib}
        } fi
        otool -L ${target_dylib}
    } done
}

patch_lib easycl . "clew"
patch_lib cocl . "easycl clblast clew"
patch_lib clblast . "clew"

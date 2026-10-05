#!/usr/bin/env bash
# Usage: bash src/test/test_parasails_exit.sh <sequential-build-dir> [--sanitize]
set -euo pipefail
source_root=$(cd "$(dirname "$0")/.." && pwd)
build=$(cd "${1:?Pass a configured and built sequential Hypre directory}" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
flags=(-std=gnu99 -fopenmp -g)
case "${2:-}" in
    "") ;;
    --sanitize) flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer) ;;
    *) printf 'Unknown option: %s\n' "$2" >&2; exit 2 ;;
esac
"${CC:-cc}" "${flags[@]}" \
    -I"$build" -I"$source_root" -I"$source_root/utilities" \
    "$source_root/test/test_parasails_exit.c" \
    "$source_root/distributed_ls/ParaSails/Mem.c" \
    -L"$build/lib" -Wl,-rpath,"$build/lib" -lHYPRE -lm \
    -o "$tmp/test_parasails_exit"
"$tmp/test_parasails_exit"

#!/usr/bin/env bash
# Compiles .tex files with latexmk.
#
# Usage:
#   ./build.sh                 build every .tex file in the repo
#   ./build.sh "Homework 3"    build only files under a given path
#   ./build.sh --clean         remove latexmk build artifacts everywhere
set -euo pipefail

if [ "${1:-}" = "--clean" ]; then
    find . -path ./.venv -prune -o -name '*.tex' -print0 |
    while IFS= read -r -d '' texfile; do
        dir=$(dirname "$texfile")
        file=$(basename "$texfile")
        (cd "$dir" && latexmk -c "$file")
    done
    exit 0
fi

search_root="${1:-.}"

find "$search_root" -path ./.venv -prune -o -name '*.tex' -print0 |
while IFS= read -r -d '' texfile; do
    dir=$(dirname "$texfile")
    file=$(basename "$texfile")
    echo "==> Building $texfile"
    (cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$file")
done

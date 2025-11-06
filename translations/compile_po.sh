#!/usr/bin/env sh

# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
fi

lang_filter="$@"

function should_compile_po() {
  lang="$1"
  if [ -z "$lang_filter" ]; then
    return 0
  fi
  for filter_lang in $lang_filter; do
    if [ "$filter_lang" == "$lang" ] ; then
      return 0
    fi
  done
  return 1
}

find . -mindepth 1 -maxdepth 1 -type f -name "*.po" -printf '%f\n' | while read po_file ; do
  lang="${po_file%.po}"
  should_compile_po $lang || continue
  lang_dir="../lada/locale/$lang/LC_MESSAGES"
  if [ ! -d "$lang_dir" ] ; then
    mkdir -p "$lang_dir"
  fi
  echo "Compiling language $lang .po file into .mo file"
  msgfmt "$po_file" -o "$lang_dir/lada.mo"
done
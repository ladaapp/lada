#!/usr/bin/env sh

# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

translations_dir=$(dirname -- "$0")
cwd_back="$(pwd)"
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
fi

lang_filter=""
for arg in "$@"; do
  case "$arg" in
    --release)
      lang_filter=$(cat release_ready_translations.txt | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      if [ -z "$lang filter" ]; then
          echo "No translations in release_ready_translations.txt"
          exit 1
      fi
      ;;
  esac
done

should_compile_po() {
  lang="$1"
  if [ -z "$lang_filter" ]; then
    return 0
  fi
  echo "$lang_filter" | tr " " "\n" | grep -q "$lang"
  return $?
}

# Clean up compiled translations if there is no corresponding .po file anymore (deleted translations)
find ../lada/locale/ -mindepth 2 -maxdepth 2 -type d -name LC_MESSAGES | while read lang_dir; do
  lang=$(basename "$(dirname "$lang_dir")")
  po_file="$lang.po"
  if [ ! -f "$po_file" ]; then
    rm -rf "$(dirname $lang_dir)"
  fi
done

# Compile .po files
find . -mindepth 1 -maxdepth 1 -type f -name "*.po" -printf '%f\n' | while read po_file ; do
  lang="${po_file%.po}"
  if ! should_compile_po $lang ; then
    _lang_dir="../lada/locale/$lang"
    [ -d "$_lang_dir" ] && rm -r "$_lang_dir"
    continue
  fi
  lang_dir="../lada/locale/$lang/LC_MESSAGES"
  if [ ! -d "$lang_dir" ] ; then
    mkdir -p "$lang_dir"
  fi
  echo "Compiling language $lang .po file into .mo file"
  msgfmt "$po_file" -o "$lang_dir/lada.mo"
done

cd "$cwd_back"
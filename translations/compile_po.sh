#!/usr/bin/env sh

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
fi

find . -mindepth 1 -maxdepth 1 -type f -name "*.po" -printf '%f\n' | while read po_file ; do
  lang="${po_file%.po}"
  lang_dir="../lada/locale/$lang/LC_MESSAGES"
  if [ ! -d "$lang_dir" ] ; then
    mkdir -p "$lang_dir"
  fi
  echo "Compiling language $lang .po file into .mo file"
  msgfmt "$po_file" -o "$lang_dir/lada.mo"
done
#!/usr/bin/env sh

# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
fi

find . -mindepth 1 -maxdepth 1 -type f -name "*.po" -printf '%f\n' | while read po_file ; do
  lang="${po_file%.po}"
  echo "Updating language $lang .po file"
  msgmerge --no-fuzzy-matching --no-wrap --previous --update "$po_file" lada.pot
done
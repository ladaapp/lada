#!/usr/bin/env sh

# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

translations_dir=$(dirname -- "$0")
if [ "$(pwd)" != "$translations_dir" ] ; then
  cd "$translations_dir"
  # go to project root so the file paths in the .pot file will show up nicely
  cd ..
fi
export TZ=UTC

echo "Updating template .pot file"
xgettext \
    --package-name=lada \
    --msgid-bugs-address=https://codeberg.org/ladaapp/lada/issues \
    --from-code=UTF-8 \
    --no-wrap \
    -f <( find lada/gui lada/cli -name "*.ui" -or -name "*.py" ) \
    -o $translations_dir/lada.pot

python3 $translations_dir/extract_csv_strings.py lada/utils/encoding_presets.csv $translations_dir/csv_strings.pot 'preset_description(translatable)'
echo "Merging extracted strings from python files with strings from encoding_presets.csv"
msgcat --no-wrap $translations_dir/lada.pot $translations_dir/csv_strings.pot -o $translations_dir/lada.pot
rm $translations_dir/csv_strings.pot

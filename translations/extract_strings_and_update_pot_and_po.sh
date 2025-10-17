#!/bin/sh

# This scripts updates the english template file .pot by pulling out all translatable strings from the app
# To start a new translation copy this file into a language-specific .mo file. For example Spanish:
#  mkdir -p translations/es/LC_MESSAGES
#  cp translations/lada.pot translations/es/LC_MESSAGES/lada.po
# Now you can edit this file and translate the english strings. Run this bash script to sync it with the code base
# an get new/changed/deleted strings.
# To test your changes just run the app. Make sure to set the environment variable LANG to the translation lange
# Example: LANG=es lada to test the GUI or LANG=es lada-cli to test translations for the CLI.

run_xgettext() {
    xgettext \
        --package-name=lada \
        --msgid-bugs-address=https://github.com/ladaapp/issues \
        --from-code=UTF-8 \
        --no-wrap \
        -f <( (find lada/gui -name "*.ui" -or -name "*.py" ; echo lada/cli/main.py) ) \
        "$@"
}

echo "update template .pot file"
run_xgettext -o translations/lada.pot

find translations -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | while read lang ; do
  echo "update language $lang .po file"
  run_xgettext -j -o translations/${lang}/LC_MESSAGES/lada.po

  echo "compile language $lang .po file in to .mo file"
  msgfmt translations/${lang}/LC_MESSAGES/lada.po -o translations/${lang}/LC_MESSAGES/lada.mo
done

Use either `compile_po.ps1` or `compile_po.sh` to compile translations.

If run without additional arguments all .po files will be compiled.

To only compile translations that should land in a release run the script with `--release` argument.

This will only consider translations contained in the file `release_ready_translations.txt` (single line, lang codes separated by spaces)

Only if quality and completeness of a translation is good enough it should be added to that file.
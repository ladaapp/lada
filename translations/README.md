## Compile translations

Use either `compile_po.ps1` or `compile_po.sh` to compile translations.

If run without additional arguments all .po files will be compiled.

To only compile translations that are expected to be shipped in the next release add the `--release` argument.


## Include / Exclude translations in a release

Packaging scripts will use the `--release` argument. So only translations listed in the file `release_ready_translations.txt` will be included.

Before doing a release check translation completeness on weblate and / or ping recent translators if they deem the quality of the translation good enough to be shipped.

The file format is: single line, lang codes separated by spaces.

## Update translations

Updating `lada.pot` will re-sync translation strings found in the codebase and make them available to translators on [Codeberg's Weblate instance](https://translate.codeberg.org/projects/lada/lada/).

Use this script for updating the .pot file:
```bash
bash translations/update_pot.sh
```

On Weblate the addon [Update PO files to match POT (msgmerge)](https://docs.weblate.org/en/weblate-5.14/admin/addons.html#update-po-files-to-match-pot-msgmerge) is active.

This means that if you commit and push an updated .pot file Weblate will automatically update all translation files (.po).

So you should be thoughtful when to update the .pot file. If you think some strings are likely to be changed soonish before the next release you should probably wait and 
update the .pot once the strings are more or less final to avoid causing unnecessary retranslation work.

> [!NOTE]
> The script `translations/update_po.sh` is currently not used as this is done by the Weblate addon as mentioned above.

If a translator does changes on Weblate it will create a Pull Request push those changes. We are also using the squash commit of Weblate, meaning
that all commits are squashed by author and language. The PR is force-pushed / updated continuously.

> [!WARNING]
> Make sure to merge this PR before the next release.
> 
> But you should also merge this PR before you are doing updated to the .pot file to avoid merge conflicts!

On the other hand, you shouldn't merge it for every change to avoid filling the git history with to many translation commits.

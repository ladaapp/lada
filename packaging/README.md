## Update dependencies

After updating release dependencies by adjusting `uv.lock` we need to update dependencies for each release distribution as well.

```shell
### Flatpak
# No need for gui extra as pycairo and pygobject dependencies are available in flatpak gnome runtime
no_base_dependencies_args="$(uv export --frozen --no-emit-project --no-hashes --no-annotate --no-header --no-editable --no-emit-package torch --no-emit-package torchvision \
  | sed 's#==.*##;s#^#--no-emit-package #' | tr '\n' ' ')"
uv export --frozen --no-default-groups --no-emit-local --format pylock.toml --no-emit-package torch --no-emit-package torchvision \
  | uv run  packaging/flatpak/convert-pylock-to-flatpak.py --output packaging/flatpak/main/python-dependencies.yaml
uv export --frozen --no-default-groups --no-emit-local --format pylock.toml --extra nvidia $no_base_dependencies_args \
  | uv run  packaging/flatpak/convert-pylock-to-flatpak.py --output packaging/flatpak/extension_nvidia/python-dependencies.yaml
uv export --frozen --no-default-groups --no-emit-local --format pylock.toml --extra intel $no_base_dependencies_args \
 | uv run  packaging/flatpak/convert-pylock-to-flatpak.py --output packaging/flatpak/extension_intel/python-dependencies.yaml
### Docker
# No need for gui extra as the docker image will only offer Lada CLI
uv export --no-default-groups --no-emit-local --format requirements.txt --extra nvidia --group docker --no-emit-package opencv-python --frozen  > packaging/docker/requirements.txt
### Windows PyInstaller
# No need for gui extra as pycairo and pygobject dependencies will be built locally via gvsbuild
uv export --no-default-groups --no-emit-local --format requirements.txt --extra nvidia --frozen > packaging/windows/requirements.txt
```

## Release a new version

#### GUI smoke tests
* Open app
* Drop a test file (no longer than a few seconds is fine)
* Open sidebar and click *Reset to factory settings* button
* Open Watch tab
* If video and audio is playing continue
* Open Export tab
* Click *Restore* button
* If restoration finishes and you can play the file by clicking the *Open in External Program" button we're good.

#### CLI smoke tests
* Run `lada-cli --input path/to/short/test/file.mp4`
* If restoration finishes and you can play the restored file in some media player continue
* Run `lada-cli --input path/to/short/test/file.mp4 --codec hevc_nvenc`
* If restoration finishes and you can play the restored file in some media player we're good.


### Release Process

> [!TIP]
> Read README.md within the subfolder for each packaging method for specific steps how to build and package each variant

- [ ] Make sure there is no pending translations PR and `release_ready_translations.txt` is up-to-date ([documentation](../translations/README.md)). Also check `Operations | Repository Maintenance` for pending changes.
- [ ] Bump version in `lada/__init__.py` (no push to origin yet)
- [ ] Write Flatpak release notes in `packaging/flatpak/main/io.github.ladaapp.lada.metainfo.xml` (no push to origin yet)
- [ ] Create Draft Release on GitHub and write release notes
- [ ] Create Draft Release on Codeberg and write release notes
- [ ] Build Docker image on Linux build machine
- [ ] Build Flatpak on Linux build machine
- [ ] Build Windows .exe on Windows build machine
- [ ] Do smoke tests. If something looks off stop and revert changes
- [ ] `git tag v<version> ; git push origin ; git push origin tag v<version>`
- [ ] Open Draft Release on GitHub and link it to git tag
- [ ] Open Draft Release on Codeberg and link it to git tag
- [ ] Create a Pull Request for flathub/io.github.ladaapp.lada and adjust *commit* and *tag* accordingly 
- [ ] Upload Windows .7z files to GitHub Draft Release and to pixeldrain.com
- [ ] Add links/description for both Windows download options in Codeberg and GitHub Draft releases
- [ ] Publish Codeberg and GitHub Releases (make them non-draft)
- [ ] Merge Flathub Pull Request
- [ ] Push Docker image to Dockerhub including (v<version> and latest tags)
- [ ] Bump version in `lada/__init__.py` by incrementing the patch version and appending the `.dev` suffix to new version (and push to origin)

## Creating `requirements.txt`

```shell
python -m venv .venv_requirements_cli
source .venv_requirements_cli/bin/activate
pip install pip-tools
test -f packaging/requirements-cli.txt && rm packaging/requirements-cli.txt 
pip-compile --extra basicvsrpp -o packaging/requirements-cli.txt setup.py
sed -i 's#opencv-python#opencv-python-headless#' packaging/requirements-cli.txt
test -f packaging/requirements-gui.txt && rm packaging/requirements-gui.txt 
pip-compile --extra basicvsrpp,gui -o packaging/requirements-gui.txt setup.py
deactivate
rm -r .venv_requirements_cli
```

TODO: Revise this process now that Windows is also part of the release train

## Release Process
1. Create GitHub PR for Flathub/Lada repository (check README there)
2. Once pipeline is green pull this flatpak from flathub
   1. Test GUI: 
        1. Open file
        2. Play/Pause file, Seek file, Change a setting in the sidebar, Reset settings in the sidebar
        3. Export file
   2. Test CLI: export file
3. Build docker image from `Dockerfile.Latest` and test CLI export with the same file
4. Assuming all looks good and nothing is broken: Push a commit in Ladaapp/lada with these changes:
    1. Bump version in `lada/__init__.py`
    2. Write short release notes in `packaging/flatpak/share/metainfo/io.github.ladaapp.lada.metainfo.xml`
    3. Update git tags in `packaging/docker/Dockerfile.Release` pointing to new git tag for this release vX.Y.Z (actual git tag will be created in 5.)
5. Create a Release on Ladaapp/Lada and add a new git tag vX.Y.Z pointing to commit from step 4). Release notes should be mostly copy-paste from what was written in the xml file but can be more verbose if necessary.
6. Update the PR in Flathub/lada repo with the release git tag and commit id
7. Once pipeline went through merge the PR on Flathub/lada. After a few hours it should be available on flathub
8. Build, tag and push new Docker image to Dockerhub built from `Dockerfile.Release` (make sure to use the same tag as was used on git, as well as 'latest')
9. Add `-dev` suffix to version in `lada/__init__.py` to start the next cycle
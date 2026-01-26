> [!NOTE]
> The manifest `io.github.ladaapp.lada.yaml` in this directory is used only for building the flatpak locally.
> 
> The manifest used for building the flatpak available on Flathub is maintained in the [flathub repo on GitHub](https://github.com/flathub/io.github.ladaapp.lada).
>
> Check out the sections below for further details.

## Build and publish to Flathub

The Flatpak on Flathub is build via CI. You just have to open (and merge) a pull request on this repository:

https://github.com/flathub/io.github.ladaapp.lada

It contains a very similar manifest file. Adjust it (in most cases just update the git tag) and create a pull request with these changes.

If the pipeline succeeds it should post a comment to the PR with a link to install the built flatpak.

Only if the PR gets merged the production pipeline runs which will push the new flatpak to Flathub.

Note, that it will take a few hours before the new Flatpak is available on Flathub. It takes even longer before the Flathub website gets updated.

```shell
uv run packaging/flatpak/convert-pylock-to-flatpak.py
```

## Build and install locally

Setup dependencies
```shell
flatpak remote-add --if-not-exists --user flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user -y flathub org.flatpak.Builder
```

Build and install:
```shell
flatpak run org.flatpak.Builder --force-clean --user --install --install-deps-from=flathub build_flatpak packaging/flatpak/main/io.github.ladaapp.lada.yaml
# Install only one of these extensions at the same time!
# Nvidia
flatpak run org.flatpak.Builder --force-clean --user --install --install-deps-from=flathub build_flatpak packaging/flatpak/extension_nvidia/io.github.ladaapp.lada.extensions.nvidia.yaml
# Intel
flatpak run org.flatpak.Builder --force-clean --user --install --install-deps-from=flathub build_flatpak packaging/flatpak/extension_intel/io.github.ladaapp.lada.extensions.intel.yaml
```

Check for linting errors
```shell
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest packaging/flatpak/main/io.github.ladaapp.lada.yaml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest packaging/flatpak/extension_nvidia/io.github.ladaapp.lada.extensions.nvidia.yaml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest packaging/flatpak/extension_intel/io.github.ladaapp.lada.extensions.intel.yaml
```

```shell
flatpak run --command=flatpak-builder-lint org.flatpak.Builder appstream packaging/flatpak/main/io.github.ladaapp.lada.metainfo.xml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder appstream packaging/flatpak/extension_nvidia/io.github.ladaapp.lada.extensions.nvidia.metainfo.xml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder appstream packaging/flatpak/extension_intel/io.github.ladaapp.lada.extensions.intel.metainfo.xml
```

Lada is now installed. You should be able to find it in your application launcher as `Lada (dev)`.

Or you run it via `flatpak run io.github.ladaapp.lada//main`.


## Update python dependencies

All python dependencies are specified in and installed via `python-dependencies.yaml` flatpak module.

This file is generated py the script `convert-pylock-to-flatpak.py` based on `uv.lock` located in the root of the project.

See packaging [README.md](../README.md) for context.
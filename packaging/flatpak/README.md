Lada is available on Flathub, check out the README in the repo linked below to find out more how to package the app via Flatpak:

https://github.com/flathub/io.github.ladaapp.lada

### Tip

Didn't have this issue before, but now as my GitHub account is borked the GitHub bot does not seem to report the .flatpakref location anymore

Here is the workaround to get this URL:

* Once the build is green
* Open Job `build-x86_64` / `Upload build`
* Search for a log like `Uploading refs to https://hub.flathub.org/api/v1/build/221353`, mark the flathub build id, here `221353`
* With this build id at hand we can now install it with flatpak `flatpak install --user https://dl.flathub.org/build-repo/<flathub build id>/io.github.ladaapp.lada.flatpakref`
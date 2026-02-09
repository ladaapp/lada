FROM python:3.13.12-slim-trixie as build
RUN useradd --create-home lada
RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext patch curl xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER lada
WORKDIR /home/lada
RUN mkdir -p .local/bin \
    && curl -L 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz' \
        | tar xJf - -C .local/bin --strip-components 2 ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffmpeg ffmpeg-n8.0-latest-linux64-gpl-8.0/bin/ffprobe \
    && chmod 555 .local/bin/*
COPY packaging/docker/requirements.txt packaging/docker/requirements.txt
RUN pip install --user --no-deps --no-cache-dir --requirement packaging/docker/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
COPY patches patches
RUN    patch -u -p1 -d .local/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch \
    && patch -u -p1 -d .local/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch \
    && patch -u -p1 -d .local/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
COPY --chown=lada:lada . .
RUN sh translations/compile_po.sh --release
RUN pip install --user --no-deps --no-cache-dir  '.'


FROM python:3.13.12-slim-trixie
RUN useradd --create-home lada
RUN mkdir -p /home/lada/.config/Ultralytics \
    && echo '{"settings_version":"0.0.6","datasets_dir":"datasets","weights_dir":"weights","runs_dir":"experiments","uuid":"dummy","sync":false,"api_key":"","openai_api_key":"","clearml":false,"comet":false,"dvc":false,"hub":false,"mlflow":false,"neptune":false,"raytune":false,"tensorboard":false,"wandb":false,"vscode_msg":false,"openvino_msg":false}' > /home/lada/.config/Ultralytics/settings.json \
    && chown -R lada:lada /home/lada/.config
RUN mkdir -p model_weights/3rd_party mnt && chmod -R 555 model_weights && chmod 777 mnt
ADD --checksum=sha256:056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5 \
    https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true model_weights/lada_mosaic_detection_model_v2.pt
ADD --checksum=sha256:c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f \
    https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true model_weights/lada_mosaic_detection_model_v4_accurate.pt
ADD --checksum=sha256:9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4 \
    https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true model_weights/lada_mosaic_detection_model_v4_fast.pt
ADD --checksum=sha256:d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee \
    https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true model_weights/lada_mosaic_restoration_model_generic_v1.2.pth
ADD --checksum=sha256:5643ca297c13920b8ffd39a0d85296e494683a69e5e8204d662653d24c582766 \
    https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t model_weights/3rd_party/clean_youknow_video.pth
RUN chmod 444 model_weights/*\.pt* model_weights/3rd_party/*\.pt*
COPY --from=build --chown=lada:lada /home/lada/.local /home/lada/.local
USER lada
ENV PATH=/home/lada/.local/bin:$PATH
ENV LOCALE_DIR=/home/lada/.local/lib/python3.13/site-packages/lada/locale

ENTRYPOINT ["lada-cli"]
CMD ["--help"]
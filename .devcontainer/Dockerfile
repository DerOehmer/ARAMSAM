FROM nvcr.io/nvidia/pytorch:24.05-py3
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev ffmpeg libsm6 libxext6 libegl1 -y
FROM nvcr.io/nvidia/pytorch:24.05-py3
COPY docker_requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/docker_requirements.txt \
    && rm -rf /tmp/pip-tmp
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev ffmpeg libsm6 libxext6 libegl1 '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
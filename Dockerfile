FROM ubuntu:14.04

RUN apt-get update
RUN apt-get install -y \
 git \
 curl \
 cmake \
 xvfb \
 xorg-dev \
 libpq-dev \
 libjpeg-dev \
 libav-tools \
 libxcursor1 \
 libxinerama1 \
 libglu1-mesa \
 libgl1-mesa-dev \
 python-numpy \
 python-scipy \
 python-pyglet \
 python-setuptools
RUN apt-get autoremove

RUN easy_install pip
RUN pip install gym
RUN pip install gym[atari]
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
RUN pip install git+http://github.com/mthrok/openai_fitness.git@feature/dqn2

CMD ["exercise", "--help"]

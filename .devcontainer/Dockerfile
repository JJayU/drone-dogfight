# ROS2 Humble base image
FROM ros:humble

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    mesa-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ros-humble-ros-base \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-ament-index-python \
    libboost-all-dev \
    libusb-1.0-0-dev \
    pkg-config \
    python3-rosdep \
    ros-humble-tf-transformations \
    ros-humble-motion-capture-tracking-interfaces \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo
RUN mkdir -p /root/.mujoco && \
    wget -qO /root/.mujoco/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf /root/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco && \
    rm /root/.mujoco/mujoco210-linux-x86_64.tar.gz

# Setup MuJoCo system variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Install Python packages
RUN pip3 install --upgrade --no-cache-dir \
    mujoco \
    mujoco-python-viewer \
    numpy==1.23.5 \
    scipy \
    setuptools \
    wheel \
    twine \
    check-wheel-contents

# Build ROS2 workspace
RUN . /opt/ros/humble/setup.sh 

# Setup ROS2 paths in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/ws/install/setup.bash" >> ~/.bashrc

# Acados install
WORKDIR /home 
RUN git clone https://github.com/acados/acados.git
WORKDIR /home/acados
RUN git submodule update --recursive --init
RUN mkdir -p build
WORKDIR /home/acados/build
RUN cmake -DACADOS_WITH_QPOASES=ON .. 
RUN make install -j4
RUN pip install -e /home/acados/interfaces/acados_template
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:'/home/acados/lib'" >> ~/.bashrc && \
    echo "export ACADOS_SOURCE_DIR='/home/acados'" >> ~/.bashrc

# Launch container in bash
CMD ["/bin/bash"]

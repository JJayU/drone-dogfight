# 📌 Bazowy obraz z ROS2 Humble
FROM ros:humble

# 📌 Instalacja wymaganych pakietów systemowych
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

# 📌 Instalacja MuJoCo
RUN mkdir -p /root/.mujoco && \
    wget -qO /root/.mujoco/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf /root/.mujoco/mujoco210-linux-x86_64.tar.gz -C /root/.mujoco && \
    rm /root/.mujoco/mujoco210-linux-x86_64.tar.gz

# 📌 Ustawienie zmiennej środowiskowej dla MuJoCo
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# 📌 Instalacja pakietów Python
RUN pip3 install --no-cache-dir \
    mujoco \
    mujoco-python-viewer \
    numpy==1.23.5 \
    scipy

RUN pip install stable-baselines3[extra] \
    pip install tensorboard

# 📌 Budowa workspace ROS2
RUN . /opt/ros/humble/setup.sh 

# 📌 Ustawienie ścieżek ROS2 w bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/ws/install/setup.bash" >> ~/.bashrc

# 📌 Uruchamiamy kontener w trybie bash
CMD ["/bin/bash"]

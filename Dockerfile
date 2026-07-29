FROM my_jetson_env:chkpt1

# Install Python dependencies required for dehazing inference and visualization.
# Pinned versions ensure reproducibility on JetPack-based environments.
RUN pip3 install --no-cache-dir \
    matplotlib==3.3.4 \
    "Pillow<9" \
    kiwisolver==1.3.1

WORKDIR /workspace

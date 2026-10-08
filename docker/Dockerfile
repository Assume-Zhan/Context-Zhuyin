ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel

# ---------------------------------------------------------------------------
# Stage 1: build libchewing from source (Rust core + C API).
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS chewing-builder

ARG LIBCHEWING_REPO=https://codeberg.org/chewing/libchewing.git
ARG LIBCHEWING_VERSION=v0.13.1
# libchewing v0.13.1 declares rust-version 1.88.0 in Cargo.toml.
ARG RUST_TOOLCHAIN=stable

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git pkg-config libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain ${RUST_TOOLCHAIN}

RUN git clone --depth 1 --branch ${LIBCHEWING_VERSION} ${LIBCHEWING_REPO} /src/libchewing \
    && cmake -S /src/libchewing -B /src/libchewing/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/libchewing \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_DOC=OFF \
        -DBUILD_TESTING=OFF \
        -DWITH_SQLITE3=ON \
    && cmake --build /src/libchewing/build -j"$(nproc)" \
    && cmake --install /src/libchewing/build \
    && mkdir -p /opt/libchewing/contrib \
    && cp -r /src/libchewing/contrib/python /opt/libchewing/contrib/python

# ---------------------------------------------------------------------------
# Stage 2: dev image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS dev

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

# The base image installs torch with pip into the system Python 3.12, which
# is marked externally managed (PEP 668). Allow pip to keep using it.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git git-lfs htop less \
        libsqlite3-0 locales pkg-config sudo tmux vim wget \
    && sed -i 's/^# *\(en_US.UTF-8\)/\1/; s/^# *\(zh_TW.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# libchewing: shared library, headers, dictionaries, chewing-cli.
COPY --from=chewing-builder /opt/libchewing /opt/libchewing
RUN echo /opt/libchewing/lib > /etc/ld.so.conf.d/libchewing.conf \
    && if [ -d /opt/libchewing/lib64 ]; then echo /opt/libchewing/lib64 >> /etc/ld.so.conf.d/libchewing.conf; fi \
    && ldconfig

# Python deps. The constraints file pins the torch build shipped with the
# base image so pip never replaces it with a different CUDA variant.
COPY requirements.txt /tmp/requirements.txt
RUN (pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|numpy)==' || true) > /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt /tmp/constraints.txt

# Non-root user matching the host UID and GID so files written into the bind
# mount are owned by the host user. Reuse existing ids if the base has them.
RUN if getent group ${GID} >/dev/null; then \
        groupmod -n ${USERNAME} "$(getent group ${GID} | cut -d: -f1)"; \
    else \
        groupadd -g ${GID} ${USERNAME}; \
    fi \
    && if getent passwd ${UID} >/dev/null; then \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m -g ${GID} "$(getent passwd ${UID} | cut -d: -f1)"; \
    else \
        useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}; \
    fi \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /cache/huggingface \
    && chown -R ${UID}:${GID} /cache

# Runtime settings. The OMP and thread settings follow the CPU budget notes
# in docs/proposal.md (avoid spin waiting thread pools).
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OMP_WAIT_POLICY=PASSIVE \
    MKL_NUM_THREADS=1 \
    CHEWING_PATH=/opt/libchewing/share/libchewing \
    PATH=/opt/libchewing/bin:/home/${USERNAME}/.local/bin:${PATH}

USER ${USERNAME}
WORKDIR /workspace

CMD ["bash"]

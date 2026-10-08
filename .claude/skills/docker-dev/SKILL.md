---
name: docker-dev
description: Build, start, and use the project dev container (docker/Dockerfile, docker/docker-compose.yml). Use when running code, installing packages, checking GPU access, or changing the Docker setup.
---

# Docker dev workflow

All Docker files live in `docker/`. Never place Dockerfiles, compose files, or
dockerignore files elsewhere in the repo.

## Commands (from repo root)

```bash
bash docker/host_setup.sh   # once per host, writes docker/.env with HOST_UID and HOST_GID
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d phonetic-candidate-dev
docker compose -f docker/docker-compose.yml exec phonetic-candidate-dev bash
docker compose -f docker/docker-compose.yml exec phonetic-candidate-dev python scripts/<name>.py
docker compose -f docker/docker-compose.yml down
```

Do not run `build` or `up` on your own initiative; print the commands for the
user unless they asked you to run them.

## Facts about the image

- Base: `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel`. RTX 50 series needs
  CUDA 12.8 or newer; do not downgrade below that.
- The base image has no conda: torch lives in the system Python 3.12
  (`/usr/local/lib/python3.12/dist-packages`), which is PEP 668 externally
  managed. `PIP_BREAK_SYSTEM_PACKAGES=1` is set in the dev stage; any new
  build stage that calls pip needs it too. Prefer apt for build tools
  (cmake comes from apt, 3.28).
- torch is pinned to the base image build through a constraints file. When
  adding packages to `docker/requirements.txt`, never add `torch` itself.
- libchewing is built from source (codeberg tag `LIBCHEWING_VERSION`) into
  `/opt/libchewing`. Dictionaries: `$CHEWING_PATH`.
- Repo is bind mounted at `/workspace`; `PYTHONPATH=/workspace/src`.
- HF cache is a named volume at `/cache/huggingface` (`HF_HOME`).
- Container user `dev` gets the host UID and GID from `docker/.env`
  (`HOST_UID`, `HOST_GID`, written by `docker/host_setup.sh`, gitignored;
  defaults to 1000 when missing). Rebuild after the ids change.
  The user has passwordless sudo.
- Claude Code CLI is installed for user `dev` (`~/.local/bin/claude`).
  `CLAUDE_CONFIG_DIR=/home/dev/.claude` is the named volume `claude-config`,
  so login and settings survive container recreation. Rebuild the image to
  get a newer CLI version that persists.
- Thread env defaults: `OMP_NUM_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`.

## Validating Docker changes without building

```bash
docker compose -f docker/docker-compose.yml config > /dev/null
docker build --check -f docker/Dockerfile docker
```

## Quick in-container checks

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import ctypes; ctypes.CDLL('libchewing.so.3')"
nvidia-smi
```

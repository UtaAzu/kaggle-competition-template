FROM mcr.microsoft.com/devcontainers/python:3.10

# Quick-fix: remove libegl1-mesa / libgl1-mesa-dri which may not be available in current Debian mirrors.
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    git \
    git-lfs \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
  && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
     > /etc/apt/sources.list.d/github-cli.list \
  && apt-get update -y \
  && apt-get install -y --no-install-recommends gh \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
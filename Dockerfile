FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive

# SteamCMD ships a 32-bit binary, so i386 has to be enabled before its runtime
# libraries can even be installed.
RUN dpkg --add-architecture i386 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      tzdata \
      python3 \
      python3-venv \
      lib32gcc-s1 \
      lib32stdc++6 \
 && rm -rf /var/lib/apt/lists/*

# SteamCMD writes into its own install directory and complains loudly when run
# as root, so it gets a dedicated unprivileged user.
RUN useradd --create-home --home-dir /home/steam --shell /bin/bash steam

# Docker does not update HOME when USER changes, and SteamCMD keeps its client
# under $HOME/Steam — left at /root it would try to write somewhere the steam
# user cannot touch.
ENV HOME=/home/steam

# A venv sidesteps Debian's PEP 668 "externally managed environment" refusal
# without needing --break-system-packages.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir Flask==3.0.3 waitress==3.0.0

# Install SteamCMD.
RUN mkdir -p /steamcmd /steamcmd/steamapps/workshop /app \
 && curl -sSL https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz \
      | tar -xz -C /steamcmd \
 && chown -R steam:steam /steamcmd /app

COPY --chown=steam:steam app.py index.html /app/

USER steam
WORKDIR /steamcmd

# First run bootstraps SteamCMD: it downloads its own updated client, then
# exits. It often returns non-zero even when that worked, so the result is
# checked by looking for the binary it should have unpacked.
RUN /steamcmd/steamcmd.sh +quit || true; \
    test -x /steamcmd/linux32/steamcmd \
      || (echo "SteamCMD bootstrap failed: /steamcmd/linux32/steamcmd missing" && exit 1)

WORKDIR /app

ENV STEAMCMD_DIR=/steamcmd \
    PORT=9976 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 9976

CMD ["waitress-serve", "--host=0.0.0.0", "--port=9976", "--threads=8", "app:app"]

# Steam Workshop Downloader

A small web app for a NAS: type any Steam **App ID** and **Workshop item ID**, and it
runs SteamCMD (`login anonymous`) in the background and hands you the downloaded files
through your browser. Nothing is hardcoded to a particular game.

Everything is driven from the web UI — after `docker compose up -d` you never need a
shell inside the container.

## Run it

```bash
docker compose up -d
```

Then open **http://<your-nas-ip>:9976**.

First build takes a few minutes: it installs the i386 runtime libraries SteamCMD needs,
downloads SteamCMD, and lets it bootstrap itself.

```bash
docker compose logs -f      # watch the build/run
docker compose down         # stop (downloads are kept)
```

## Using it

1. **App ID** — the game's numeric Steam ID. It is the number in the store URL:
   `store.steampowered.com/app/431960/` → `431960`.
2. **Workshop item** — either the bare numeric ID or the whole URL pasted straight from
   the browser; the ID is parsed out of
   `https://steamcommunity.com/sharedfiles/filedetails/?id=123456789`.
3. **Download** — SteamCMD's output streams into the log panel. When it finishes, every
   downloaded file is listed with its own download link.

Downloads run **one at a time**. SteamCMD keeps mutable state in its own install
directory and does not tolerate two copies running against it, so a second request waits
its turn instead of corrupting the first.

## Things that are Steam's limits, not bugs in this app

- **Removed / taken-down items usually stay unavailable.** If Steam has pulled a Workshop
  item, anonymous login will almost certainly still fail to fetch it. There is no
  workaround here — the content is gone from the CDN's reach for anonymous clients.
- **Some items require a real Steam account.** Depending on the creator's privacy setting
  or the game's DRM, an item may be unreachable with `login anonymous`, which is the only
  mode this app uses. Items marked private or friends-only fall in this group.
- **A wrong App ID looks exactly like a removed item.** SteamCMD reports the same generic
  failure either way, so double-check the App ID actually owns the item before concluding
  it was taken down.

When a download produces no files the app reads SteamCMD's log and tells you which of
these it most likely was, rather than silently reporting success.

## Where files live

Inside the container:

```
/steamcmd/steamapps/workshop/content/<appid>/<itemid>/
```

That whole `workshop` tree is on the named volume `workshop-content`, so downloads
survive restarts and image rebuilds.

To copy something out without the browser:

```bash
docker compose cp workshop-downloader:/steamcmd/steamapps/workshop/content/431960/450814997 ./out
```

To wipe everything downloaded:

```bash
docker compose down -v
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/download` | `{appid, itemid}` → starts a background job, returns `{job_id}` (202). `itemid` may be a full Workshop URL. |
| `GET` | `/api/status/<job_id>` | `{status, message, log, next_index, elapsed}`. `status` is `queued`/`running`/`done`/`failed`. Pass `?from=<next_index>` to fetch only new log lines. |
| `GET` | `/api/files/<appid>/<itemid>` | Lists downloaded files with sizes and download URLs. |
| `GET` | `/api/get/<appid>/<itemid>/<filename>` | Serves one file as an attachment. Nested paths are supported; anything escaping the item directory is refused. |
| `GET` | `/api/health` | Liveness plus whether the SteamCMD binary is present. |

## Configuration

Set these in `docker-compose.yml` under `environment:`.

| Variable | Default | Meaning |
|---|---|---|
| `JOB_TIMEOUT` | `3600` | Seconds before a wedged SteamCMD run is killed. |
| `MAX_LOG_LINES` | `3000` | Log lines kept per job. |
| `MAX_JOBS` | `50` | Job records kept in memory. |
| `PORT` | `9976` | Port inside the container. |
| `STEAMCMD_DIR` | `/steamcmd` | SteamCMD install directory. |
| `WORKSHOP_ROOT` | `$STEAMCMD_DIR/steamapps/workshop/content` | Where downloads land. |

## Running without Docker

```bash
python3 -m venv venv && ./venv/bin/pip install Flask waitress
STEAMCMD_DIR=/path/to/steamcmd ./venv/bin/waitress-serve --port=9976 app:app
```

## Notes on the container

- Base image is `debian:bookworm` with `dpkg --add-architecture i386`, because SteamCMD
  is a 32-bit binary and needs `lib32gcc-s1` and `lib32stdc++6`.
- Python packages live in a venv at `/opt/venv`, which avoids Debian's PEP 668
  "externally managed environment" refusal without `--break-system-packages`.
- SteamCMD runs as an unprivileged `steam` user, and `HOME` is set to `/home/steam` so it
  can write its client files.
- The image build fails loudly if SteamCMD's self-bootstrap did not produce
  `/steamcmd/linux32/steamcmd`, rather than shipping a broken image.

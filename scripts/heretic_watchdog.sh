#!/usr/bin/env bash
# Heretic study watchdog.
#
# WHY THIS EXISTS
# ---------------
# A previous run reached 9h30m and then died when the vLLM engine restarted:
# the shim logged `ConnectionRefusedError` and `Remote end closed connection
# without response`, every request failed, and the study had to be relaunched by
# hand from a one-off `bash -c` line. Neither the study nor the shim is
# supervised -- both are orphans parented to PID 1 -- so a crash at hour 20 of a
# 33-hour run simply ends the study with no result.
#
# WHAT IT DOES
# ------------
# Every INTERVAL seconds, confirm the engine is up, then confirm the shim and
# the study are alive, and restart whichever is missing. It never touches a
# running study.
#
# It deliberately does NOT restart the study while the engine is unhealthy: a
# new study launched into a dead engine would burn Optuna trials on guaranteed
# failures, which is worse than waiting.
#
# Install and run:
#   sudo cp heretic_watchdog.sh /usr/local/bin/heretic-watchdog
#   sudo chmod +x /usr/local/bin/heretic-watchdog
#   sudo setsid nohup /usr/local/bin/heretic-watchdog >/dev/null 2>&1 &
#
# Stop with:  sudo pkill -f heretic-watchdog

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN=/opt/heretic-dgx
RUN_USER=hyudryu
STUDY_LOG="$RUN/prod-run1.log"
WATCH_LOG="$RUN/watchdog.log"
SHIM=/home/hyudryu/heretic_shim.py
SHIM_LOG=/home/hyudryu/heretic_shim.log
SHIM_UPSTREAM=http://127.0.0.1:8014
ENGINE_HEALTH=http://127.0.0.1:8014/health
SHIM_HEALTH=http://127.0.0.1:8765/health
SEED=20250913
INTERVAL=60
# A restart must not be attempted more often than this, so a genuinely broken
# study cannot be hammered into a fork bomb.
COOLDOWN=300
LOCK=/run/heretic-watchdog.lock

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$WATCH_LOG"; }

http_ok() {
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null)" = "200" ]
}

# The patterns are ANCHORED to the interpreter path deliberately.
#
# A loose pattern like 'bin/heretic --model' also matches the launcher shell,
# whose own command line contains the entire heretic invocation. That shell can
# outlive the study, so a loose pattern would report "study alive" forever and
# the watchdog would never restart anything. Verified against the live system:
#   'bin/heretic --model /models/DeepSeek-V4'          -> 2377625 2377627  (bash + study)
#   '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' -> 2377627      (study only)
study_pid() { pgrep -f '^/opt/heretic-dgx/.venv/bin/python .venv/bin/heretic' 2>/dev/null | head -1; }
shim_pid() { pgrep -f '^/opt/heretic-dgx/.venv/bin/python /home/hyudryu/heretic_shim.py' 2>/dev/null | head -1; }

start_shim() {
    say "shim missing -- starting it (upstream $SHIM_UPSTREAM)"
    HERETIC_SHIM_UPSTREAM="$SHIM_UPSTREAM" setsid nohup \
        /opt/heretic-dgx/.venv/bin/python "$SHIM" \
        >>"$SHIM_LOG" 2>&1 &
    sleep 5
    if [ -n "$(shim_pid)" ]; then
        say "shim started (pid $(shim_pid))"
    else
        say "WARN shim failed to start; see $SHIM_LOG"
    fi
}

start_study() {
    say "study missing -- restarting from checkpoint (checkpoint_action=continue)"
    # A previous run leaves the same adapter name loaded; the shim also unloads
    # before loading, so this is belt and braces.
    curl -s -m 60 -X POST http://127.0.0.1:8765/v1/unload_lora_adapter \
        -H 'Content-Type: application/json' \
        -d '{"lora_name":"heretic-trial"}' >/dev/null 2>&1
    runuser -u "$RUN_USER" -- bash -c \
        "cd $RUN && CUDA_VISIBLE_DEVICES= nohup .venv/bin/heretic \
         --model /models/DeepSeek-V4.1-Flash --config ./config.dsv41.toml \
         --seed $SEED </dev/null >> $STUDY_LOG 2>&1 &"
    sleep 10
    if [ -n "$(study_pid)" ]; then
        say "study restarted (pid $(study_pid))"
    else
        say "WARN study failed to start; see $STUDY_LOG"
    fi
}

# --- single instance ------------------------------------------------------
exec 9>"$LOCK" 2>/dev/null || { echo "cannot open $LOCK" >&2; exit 1; }
if ! flock -n 9; then
    echo "another watchdog already holds $LOCK; exiting" >&2
    exit 0
fi

say "watchdog started (pid $$, interval ${INTERVAL}s, cooldown ${COOLDOWN}s)"
engine_down_since=""
last_study_restart=0

while true; do
    if ! http_ok "$ENGINE_HEALTH"; then
        [ -z "$engine_down_since" ] && engine_down_since=$(date +%s)
        down_for=$(( $(date +%s) - engine_down_since ))
        # Log the transition once, then stay quiet until it recovers.
        [ "$down_for" -lt $((INTERVAL + 5)) ] && \
            say "engine at $ENGINE_HEALTH is DOWN -- holding off on restarts"
    else
        if [ -n "$engine_down_since" ]; then
            down_for=$(( $(date +%s) - engine_down_since ))
            say "engine recovered after ${down_for}s"
            engine_down_since=""
        fi

        # Shim first: without it every study request 502s, and Optuna would
        # record a string of failed trials.
        if [ -z "$(shim_pid)" ]; then
            start_shim
        elif ! http_ok "$SHIM_HEALTH"; then
            say "WARN shim process alive but $SHIM_HEALTH unhealthy"
        fi

        if [ -z "$(study_pid)" ]; then
            now=$(date +%s)
            if [ $(( now - last_study_restart )) -ge "$COOLDOWN" ]; then
                last_study_restart=$now
                start_study
            else
                say "study missing but within cooldown; waiting"
            fi
        fi
    fi

    sleep "$INTERVAL"
done

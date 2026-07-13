#!/usr/bin/env bash
# ops.sh — standalone service manager for trading-bot
# Usage:
#   bash ops.sh start [micro_poller|model_server|paper_trader|paper_trader_multi|all]
#   bash ops.sh stop  [micro_poller|model_server|paper_trader|paper_trader_multi|all]
#   bash ops.sh status [name...]
#   bash ops.sh logs  <name> [lines]
#   bash ops.sh restart <name|all>
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$BASE_DIR/run"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

declare -A PID_FILE
PID_FILE=(
  [micro_poller]="$PID_DIR/micro_poller.pid"
  [model_server]="$PID_DIR/model_server.pid"
  [paper_trader]="$PID_DIR/paper_trader.pid"
  [paper_trader_multi]="$PID_DIR/paper_trader_multi.pid"
)

is_running() {
  local name=$1
  local pid_file="${PID_FILE[$name]}"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    else
      rm -f "$pid_file"
      return 1
    fi
  fi
  return 1
}

svc_start() {
  local name=$1
  if is_running "$name"; then
    echo "[$name] already running (pid $(<${PID_FILE[$name]}))"
    return 0
  fi
  echo "[$name] starting..."

  if [[ "$name" == "paper_trader_multi" ]]; then
    # 1D chart → run once per day. All runs go to a single log file.
    nohup env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" bash --norc --noprofile -c '
      cd "'"$BASE_DIR"'"
      while true; do
        PYTHONUNBUFFERED=1 "'"$BASE_DIR"'/.venv/bin/python3" -u paper_trader_multi.py 2>&1
        sleep 86400
      done
    ' >> "$LOG_DIR/${name}.log" 2>&1 < /dev/null &
  else
    nohup env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" bash --norc --noprofile -c \
      "cd '$BASE_DIR' && exec '$BASE_DIR/.venv/bin/python3' -u $name.py" \
      >> "$LOG_DIR/${name}.log" 2>&1 < /dev/null &
  fi

  local pid=$!
  echo "$pid" > "${PID_FILE[$name]}"
  echo "[$name] started pid=$pid"
}

svc_stop() {
  local name=$1
  if ! is_running "$name"; then
    echo "[$name] not running"
    return 0
  fi
  local pid
  pid=$(<"${PID_FILE[$name]}")
  echo "[$name] stopping pid=$pid..."
  kill "$pid" 2>/dev/null || true
  for i in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "${PID_FILE[$name]}"
      echo "[$name] stopped"
      return 0
    fi
    sleep 0.1
  done
  echo "[$name] did not exit, sending SIGKILL"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "${PID_FILE[$name]}"
}

svc_status() {
  local name=$1
  if is_running "$name"; then
    local pid
    pid=$(<"${PID_FILE[$name]}")
    echo "[$name] RUNNING pid=$pid log=$LOG_DIR/${name}.log"
  else
    echo "[$name] STOPPED"
  fi
}

svc_logs() {
  local name=$1
  local lines=${2:-50}
  if [[ "$name" == "paper_trader_multi" ]]; then
    # Show the most recent daily log file
    LATEST=$(ls -t "$LOG_DIR"/paper_trader_multi-*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
      echo "Showing: $LATEST"
      tail -n "$lines" "$LATEST"
    else
      echo "No daily logs found for paper_trader_multi yet"
    fi
  elif [[ -f "$LOG_DIR/${name}.log" ]]; then
    tail -n "$lines" "$LOG_DIR/${name}.log"
  else
    echo "No log for $name"
  fi
}

cmd=${1:-}
case "$cmd" in
  start)
    target=${2:-all}
    if [[ "$target" == "all" ]]; then
      svc_start micro_poller
      svc_start model_server
      svc_start paper_trader
      svc_start paper_trader_multi
    else
      svc_start "$target"
    fi
    ;;
  stop)
    target=${2:-all}
    if [[ "$target" == "all" ]]; then
      svc_stop paper_trader_multi
      svc_stop paper_trader
      svc_stop model_server
      svc_stop micro_poller
    else
      svc_stop "$target"
    fi
    ;;
  restart)
    target=${2:-all}
    if [[ "$target" == "all" ]]; then
      svc_stop paper_trader_multi
      svc_stop paper_trader
      svc_stop model_server
      svc_stop micro_poller
      sleep 1
      svc_start micro_poller
      svc_start model_server
      svc_start paper_trader
      svc_start paper_trader_multi
    else
      svc_stop "$target"
      sleep 0.5
      svc_start "$target"
    fi
    ;;
  status)
    if [[ $# -gt 1 ]]; then
      for n in "${@:2}"; do svc_status "$n"; done
    else
      svc_status micro_poller
      svc_status model_server
      svc_status paper_trader
      svc_status paper_trader_multi
    fi
    ;;
  logs)
    if [[ $# -lt 2 ]]; then
      echo "Usage: $0 logs <micro_poller|model_server|paper_trader|paper_trader_multi> [lines]"
      exit 1
    fi
    svc_logs "${@:2}"
    ;;
  *)
    cat <<EOF
Usage:
  bash ops.sh start  [micro_poller|model_server|paper_trader|paper_trader_multi|all]
  bash ops.sh stop   [micro_poller|model_server|paper_trader|paper_trader_multi|all]
  bash ops.sh restart [micro_poller|model_server|paper_trader|paper_trader_multi|all]
  bash ops.sh status [name...]
  bash ops.sh logs  <micro_poller|model_server|paper_trader|paper_trader_multi> [lines]
EOF
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROUTE_RULE='to 192.168.3.0/24 lookup main'

if ! ip link show tunx >/dev/null 2>&1; then
    printf 'tunx is not present; leaving registry routing unchanged\n'
    exit 0
fi

if ip rule show | rg --quiet 'to 192\.168\.3\.0/24 lookup main'; then
    exit 0
fi

printf 'adding runtime route rule for the private registry subnet\n'
if command -v systemd-run >/dev/null; then
    systemd-run --wait --pipe /usr/sbin/ip rule add pref 8999 to 192.168.3.0/24 lookup main
else
    sudo /usr/sbin/ip rule add pref 8999 to 192.168.3.0/24 lookup main
fi

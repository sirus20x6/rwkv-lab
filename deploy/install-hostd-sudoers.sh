#!/usr/bin/env bash
# Installs the narrow sudoers grant the TrainVM deployment needs.
#
# The host authority itself is unprivileged, so this grant covers only the
# things that genuinely require root: writing binaries into /usr/local, writing
# unit files into /etc/systemd/system, and driving systemctl for the three
# units. The hostd configuration and its GPU authorization document are owned
# and written by the unprivileged authority in its own state directory, so no
# rule here installs, reads, or authorizes them.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec /usr/bin/sudo /usr/bin/bash "$0" "$@"
fi

target=/etc/sudoers.d/trainvm-hostd
temporary=$(/usr/bin/mktemp /etc/sudoers.d/.trainvm-hostd.XXXXXX)
trap '/usr/bin/rm -f "$temporary"' EXIT

/usr/bin/cat >"$temporary" <<'SUDOERS'
Cmnd_Alias TRAINVM_HOSTD = \
    /usr/bin/install -o root -g root -m 0755 /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-hostd /usr/local/sbin/trainvm-hostd, \
    /usr/bin/install -o root -g root -m 0755 /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-gpu-fault-observer /usr/local/sbin/trainvm-gpu-fault-observer, \
    /usr/bin/install -o root -g root -m 0755 /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm /usr/local/bin/trainvm, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla-parity-integration/deploy/trainvm-hostd.service /etc/systemd/system/trainvm-hostd.service, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla-parity-integration/deploy/trainvm-gpu-fault-observer.service /etc/systemd/system/trainvm-gpu-fault-observer.service, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla-parity-integration/deploy/trainvm-controller.service /etc/systemd/system/trainvm-controller.service, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla/runs/trainvm-worker-deployment-26fe458/adapters.json /etc/trainvm/adapters.json, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla/runs/trainvm-worker-deployment-26fe458/host-launches.json /etc/trainvm/host-launches.json, \
    /usr/bin/install -o root -g root -m 0644 /thearray/git/moe-mla-parity-integration/docs/experiment-vm/examples/training-components.v1.json /etc/trainvm/training-components.json, \
    /usr/bin/install -d -o root -g webteam -m 0750 /etc/trainvm, \
    /usr/bin/systemctl daemon-reload, \
    /usr/bin/systemctl enable --now trainvm-gpu-fault-observer.service, \
    /usr/bin/systemctl enable --now trainvm-controller.service, \
    /usr/bin/systemctl enable --now trainvm-hostd.service, \
    /usr/bin/systemctl start trainvm-hostd.service, \
    /usr/bin/systemctl stop trainvm-hostd.service, \
    /usr/bin/systemctl restart trainvm-hostd.service, \
    /usr/bin/systemctl restart trainvm-controller.service, \
    /usr/bin/systemctl stop trainvm-gpu-fault-observer.service, \
    /usr/bin/systemctl restart trainvm-gpu-fault-observer.service, \
    /usr/bin/systemctl status trainvm-gpu-fault-observer.service, \
    /usr/bin/systemctl status trainvm-hostd.service, \
    /usr/bin/systemctl status trainvm-controller.service, \
    /usr/bin/systemctl is-active trainvm-gpu-fault-observer.service, \
    /usr/bin/systemctl is-active trainvm-hostd.service, \
    /usr/bin/systemctl is-active trainvm-controller.service

sirus ALL=(root) NOPASSWD: TRAINVM_HOSTD
SUDOERS

/usr/bin/chmod 0440 "$temporary"
/usr/bin/visudo -cf "$temporary"
/usr/bin/mv -f "$temporary" "$target"
trap - EXIT

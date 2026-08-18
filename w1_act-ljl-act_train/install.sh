#!/usr/bin/env bash
set -euo pipefail

CURDIR="$(pwd)"
PROJECT_DIR="$(basename "${CURDIR}")"
PACKAGE_NAME="${PROJECT_DIR//_/-}"
ARCHITECTURE="${ARCHITECTURE:-arm64}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/home/dexforce/w1/w1_act}"
VERSION="$(python3 -c 'import xml.etree.ElementTree as ET; print(ET.parse("package.xml").getroot().findtext("version"))')"
DEB_ROOT="${CURDIR}/debian/${PACKAGE_NAME}"
DEBIAN_DIR="${DEB_ROOT}/DEBIAN"

echo "Running file installation for ${PACKAGE_NAME} ${VERSION}..."

rm -rf "${DEB_ROOT}"
mkdir -p "${DEBIAN_DIR}" "${DEB_ROOT}${INSTALL_PREFIX}" "${DEB_ROOT}/etc/default"

install -m 0644 "${CURDIR}/packaging/w1-act.default" "${DEB_ROOT}/etc/default/w1-act"

tar -C "${CURDIR}" \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./debian' \
    --exclude='./dist' \
    --exclude='./release' \
    --exclude='./deb-contents.txt' \
    --exclude='*/__pycache__' \
    --exclude='*/__MACOSX' \
    --exclude='._*' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='./w1_lerobot/log' \
    --exclude='./w1_lerobot/src/lerobot.egg-info' \
    -cf - . | tar -C "${DEB_ROOT}${INSTALL_PREFIX}" -xf -

find "${DEB_ROOT}${INSTALL_PREFIX}" -type f -name '*.sh' -exec chmod 0755 {} +

cat > "${DEBIAN_DIR}/conffiles" <<EOF
/etc/default/w1-act
EOF

cat > "${DEBIAN_DIR}/postinst" <<'EOF'
#!/bin/sh
set -e
if id dexforce >/dev/null 2>&1; then
    chown -R dexforce:dexforce /home/dexforce/w1/w1_act
fi
EOF
chmod 0755 "${DEBIAN_DIR}/postinst"

cat > "${DEBIAN_DIR}/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: misc
Priority: optional
Architecture: ${ARCHITECTURE}
Maintainer: Dexforce <dexforce@dexforce.com>
Description: W1 ACT inference application
 Code, scripts, configuration, and assets for W1 ACT policy inference.
EOF

find "${DEBIAN_DIR}" -type f ! -name postinst -exec chmod 0644 {} +

echo "File installation completed."

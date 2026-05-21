#!/usr/bin/env sh
# Cortyx installer — downloads the latest pre-built binary for your platform.
# Usage: curl -fsSL https://raw.githubusercontent.com/cortyx-ai/cortyx/main/install.sh | sh
#
# Options (via env vars):
#   CORTYX_VERSION  — specific version to install (default: latest)
#   CORTYX_BIN_DIR  — install directory (default: ~/.local/bin, or /usr/local/bin if writable)

set -eu

REPO="sorunokoe/Cortyx"
BIN_NAME="cortyx"

# ─── Detect platform ─────────────────────────────────────────────────────────

detect_target() {
  OS=$(uname -s)
  ARCH=$(uname -m)

  case "$OS" in
    Linux*)
      case "$ARCH" in
        x86_64)  echo "x86_64-unknown-linux-gnu" ;;
        aarch64) echo "aarch64-unknown-linux-gnu" ;;
        *)       echo "unsupported-linux-$ARCH" ;;
      esac ;;
    Darwin*)
      case "$ARCH" in
        x86_64)  echo "x86_64-apple-darwin" ;;
        arm64)   echo "aarch64-apple-darwin" ;;
        *)       echo "unsupported-darwin-$ARCH" ;;
      esac ;;
    MINGW*|MSYS*|CYGWIN*)
      echo "x86_64-pc-windows-msvc" ;;
    *)
      echo "unsupported-$OS-$ARCH" ;;
  esac
}

# ─── Detect install dir ───────────────────────────────────────────────────────

detect_bin_dir() {
  if [ -n "${CORTYX_BIN_DIR:-}" ]; then
    echo "$CORTYX_BIN_DIR"
  elif [ -w "/usr/local/bin" ]; then
    echo "/usr/local/bin"
  else
    echo "$HOME/.local/bin"
  fi
}

# ─── Fetch latest version ─────────────────────────────────────────────────────

fetch_latest_version() {
  if command -v curl > /dev/null 2>&1; then
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\(.*\)".*/\1/'
  elif command -v wget > /dev/null 2>&1; then
    wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\(.*\)".*/\1/'
  else
    echo "ERROR: curl or wget is required" >&2
    exit 1
  fi
}

# ─── Download and install ─────────────────────────────────────────────────────

main() {
  TARGET=$(detect_target)
  if echo "$TARGET" | grep -q "^unsupported"; then
    echo "ERROR: Unsupported platform: $TARGET" >&2
    echo "Please build from source: cargo install cortyx" >&2
    exit 1
  fi

  VERSION="${CORTYX_VERSION:-$(fetch_latest_version)}"
  if [ -z "$VERSION" ]; then
    echo "ERROR: Could not determine latest version. Pass CORTYX_VERSION=vX.Y.Z" >&2
    exit 1
  fi

  BIN_DIR=$(detect_bin_dir)
  mkdir -p "$BIN_DIR"

  # Windows uses .zip; everything else .tar.gz
  case "$TARGET" in
    *windows*) EXT="zip" ;;
    *)         EXT="tar.gz" ;;
  esac

  ASSET="${BIN_NAME}-${VERSION}-${TARGET}.${EXT}"

  # B1: prefer the embed-enabled binary (hybrid BM25 + dense retrieval).
  # Set CORTYX_NO_EMBED=1 to force the plain BM25-only binary (~6MB vs ~8MB).
  if [ -z "${CORTYX_NO_EMBED:-}" ]; then
    EMBED_ASSET="${BIN_NAME}-${VERSION}-${TARGET}-embed.${EXT}"
    EMBED_URL="https://github.com/${REPO}/releases/download/${VERSION}/${EMBED_ASSET}"
    # Probe with a HEAD request; fall back to plain binary if embed variant absent.
    if command -v curl > /dev/null 2>&1; then
      if curl -fsSLI "$EMBED_URL" > /dev/null 2>&1; then
        ASSET="$EMBED_ASSET"
        echo "  (embed variant available — using hybrid BM25 + dense retrieval build)"
        echo "  Note: ~80MB embedding model downloads on first 'cortyx compile'"
        echo "  Set CORTYX_NO_EMBED=1 to install the plain BM25-only binary instead."
      fi
    fi
  fi

  URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

  echo "Installing Cortyx ${VERSION} for ${TARGET}..."
  echo "  Downloading: ${URL}"

  TMP_DIR=$(mktemp -d)
  trap 'rm -rf "$TMP_DIR"' EXIT

  if command -v curl > /dev/null 2>&1; then
    curl -fsSL "$URL" -o "$TMP_DIR/$ASSET"
  else
    wget -qO "$TMP_DIR/$ASSET" "$URL"
  fi

  # Extract
  cd "$TMP_DIR"
  case "$EXT" in
    tar.gz)
      tar xzf "$ASSET"
      ;;
    zip)
      if command -v unzip > /dev/null 2>&1; then
        unzip -q "$ASSET"
      else
        echo "ERROR: unzip is required on Windows" >&2
        exit 1
      fi
      ;;
  esac

  # Install binary
  BIN_SRC="$TMP_DIR/${BIN_NAME}"
  if [ "$EXT" = "zip" ]; then
    BIN_SRC="$TMP_DIR/${BIN_NAME}.exe"
  fi

  if [ ! -f "$BIN_SRC" ]; then
    echo "ERROR: Binary not found after extraction. Asset contents:" >&2
    ls -la "$TMP_DIR" >&2
    exit 1
  fi

  cp "$BIN_SRC" "$BIN_DIR/${BIN_NAME}"
  chmod +x "$BIN_DIR/${BIN_NAME}"

  echo ""
  echo "✓ Cortyx ${VERSION} installed to ${BIN_DIR}/${BIN_NAME}"

  # Verify installation
  if "$BIN_DIR/${BIN_NAME}" --version > /dev/null 2>&1; then
    echo "  Version: $("$BIN_DIR/${BIN_NAME}" --version)"
  fi

  # PATH hint
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
      echo ""
      echo "  Add to PATH (if not already):"
      echo "    export PATH=\"$BIN_DIR:\$PATH\""
      ;;
  esac

  echo ""
  echo "  Quick start:"
  echo "    cd your-project"
  echo "    cortyx install     # configure Claude Code / Cursor / Windsurf"
  echo "    cortyx compile .   # index your codebase"
  echo "    cortyx serve       # start the MCP server"
}

main "$@"

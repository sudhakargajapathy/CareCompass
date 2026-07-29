#!/usr/bin/env sh
set -e

PORT="${STREAMLIT_SERVER_PORT:-7860}"
ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

# Hugging Face Spaces mounts durable disk at /data when persistent storage is
# enabled. Use it for state that is expensive to rebuild, unless the operator
# has pinned the paths explicitly. Without /data these stay unset and the app
# falls back to its in-workdir defaults (./chroma_db, ./logs/audit.log).
if [ -d /data ] && [ -w /data ]; then
  : "${CHROMA_PERSIST_DIRECTORY:=/data/chroma_db}"
  : "${AUDIT_LOG_PATH:=/data/logs/audit.log}"
  export CHROMA_PERSIST_DIRECTORY AUDIT_LOG_PATH
  mkdir -p "${CHROMA_PERSIST_DIRECTORY}" "$(dirname "${AUDIT_LOG_PATH}")"
fi

ARGS="--server.port=${PORT} --server.address=${ADDRESS}"

if [ -n "${STREAMLIT_SERVER_SSL_CERT_FILE}" ] && [ -n "${STREAMLIT_SERVER_SSL_KEY_FILE}" ]; then
  ARGS="$ARGS --server.sslCertFile=${STREAMLIT_SERVER_SSL_CERT_FILE} --server.sslKeyFile=${STREAMLIT_SERVER_SSL_KEY_FILE}"
fi

exec streamlit run app.py ${ARGS}

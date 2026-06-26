#!/usr/bin/env sh
set -e

PORT="${STREAMLIT_SERVER_PORT:-7860}"
ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

ARGS="--server.port=${PORT} --server.address=${ADDRESS}"

if [ -n "${STREAMLIT_SERVER_SSL_CERT_FILE}" ] && [ -n "${STREAMLIT_SERVER_SSL_KEY_FILE}" ]; then
  ARGS="$ARGS --server.sslCertFile=${STREAMLIT_SERVER_SSL_CERT_FILE} --server.sslKeyFile=${STREAMLIT_SERVER_SSL_KEY_FILE}"
fi

exec streamlit run app.py ${ARGS}

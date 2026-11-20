#!/bin/bash

# 1. Ensure the local directory exists
mkdir -p data/remote_logs/

# 2. Download all contents from the remote "data" folder in the "tb_logs" volume 
# to the local "data/remote_logs/" directory.
uv run modal volume get  --force tb_logs data data/remote_logs/ && \
uv run modal volume rm -r tb_logs data && \
echo "Done! Logs downloaded and removed from remote volume."
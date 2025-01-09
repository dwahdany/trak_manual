#!/bin/bash
uv pip install -r requirements.txt
uv pip install fast-jl --no-build-isolation
uv pip install traker[fast]

#!/bin/bash
uv sync 
uv pip install fast-jl --no-build-isolation
uv pip install traker[fast]

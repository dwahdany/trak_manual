## Installation
### Install pixi if not present:
```shell
curl -fsSL https://pixi.sh/install.sh | bash
```
Restart terminal after installation
```shell
exec $SHELL
```

### Install dependencies:
```shell
pixi i
```
Install traker[fast]: (doesnt work through pixi 🙁)
```shell
pixi run pip install traker[fast]
```

## Running
```shell
pixi run python main.py total_worker=1000 worker_id=0
```

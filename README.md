Dependencies
------------
```bash
$ pip install -r requirements.txt
```

```bash
# follow https://docs.dgl.ai/en/0.4.x/install/ to install dgl with gpu
# e.g.
pip install dgl-cu101
```

# Run experiment
``` bash
# Copy features.csv, labels.csv and network.csv to the data directory
# Start the experiment (36 experiments in total)
sh run_experiments.sh
```
## network.csv:
```
srcId,dstId,timestamp(YYYYMMDDMISS),money,type
...
```

## features.csv
```
nodeId, age, count, ...
...
```

## labels.txt
```
nodeId,label
...
```

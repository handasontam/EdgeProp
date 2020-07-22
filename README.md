Dependencies
------------
```bash
$ pip install -r requirements.txt
```

```bash
# follow https://docs.dgl.ai/en/0.4.x/install/ to install dgl with gpu
# e.g.
$ pip install dgl-cu101
```

# Run experiment (EdgeProp)
``` bash
# Copy features.csv, labels.csv and network.csv to the data directory
# preprocess the dataa
$ python preprocess.py --data-dir ./data
# Start the experiment (120 experiments in total)
$ python run.py
```

# Run experiment (Classical Baselines)
``` bash
$ python main.py --data-dir ./data --model-dir experiments/classical_baseline/ --gpu 0
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

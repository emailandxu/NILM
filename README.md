
## convert to h5
```bash
python3 -m nilm.util dataset/redd low_freq
```


## Combinational Optimization

```bash
# redd dataset combinational optimization
python3 -m nilm.cp.main
```

## Neural Network
```bash
python3 -m nilm.nn.main
```

## Dataset tree view

```
dataset
├── redd
│   ├── high_freq_raw.tar
│   ├── high_freq.tar.bz2
│   ├── low_freq
│   │   ├── house_1
│   │   ├── house_2
│   │   ├── house_3
│   │   ├── house_4
│   │   ├── house_5
│   │   ├── house_6
│   │   └── redd_low.h5
│   ├── low_freq.tar.bz2
│   └── readme.txt
└── ukdale
    ├── UKData
    ├── ukdale.h5
    └── ukdale.h5.tgz
```
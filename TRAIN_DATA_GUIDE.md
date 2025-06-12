


```bash
#查到一些没用的图
cd data/train
find . -type f -name 'n*_*.jpg'

#实际删除
find . -type f -name 'n*_*.jpg' -exec rm -f {} +
```
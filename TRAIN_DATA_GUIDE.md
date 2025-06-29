


```bash
zip -r -s 100m image.zip data/
cat image.z* > full.zip
unzip full.zip



#查到一些没用的图
cd data
find . -type f -name 'n*_*.jpg'
find . -type f -name '*_trans_*'
find . -type f -name '*_norm_*'

#实际删除
find . -type f -name 'generate*_*.jpg' -exec rm -f {} +
```
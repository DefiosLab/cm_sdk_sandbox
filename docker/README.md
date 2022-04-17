# CM_SDK_Docker

C for Metal SDKの実行環境を整えるDockerファイル．　　

## Installation
```bash
sudo docker build ./ -t cm_sdk:1
```

`run.sh`内の`IMG`にimage名を書く．  
```bash
IMG=cm_sdk:1
```

## Run&Kill
### コンテナをたてる  
`home`フォルダと`/dev/dri`フォルダをコンテナ<->ホスト間で共有します．  
X Window Systemも共有します．  
```bash
$ ./run.sh
Container Name:
# コンテナ名を命名する
```

### コンテナを消す
```bash
$ ./clean.sh 
Remove CM!!
Container Name:
# Killしたいコンテナ名を書く
```


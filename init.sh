mkdir -p teacher_model
cd teacher_model
wget http://data.dmlc.ml/mxnet/models/imagenet/resnet/101-layers/resnet-101-0000.params ./
wget http://data.dmlc.ml/mxnet/models/imagenet/resnet/101-layers/resnet-101-symbol.json ./
cd ..
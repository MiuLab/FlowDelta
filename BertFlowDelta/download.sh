#!/usr/bin/env bash

# Download QuAC
mkdir -p QuAC_data
wget https://s3.amazonaws.com/my89public/quac/train.json -O QuAC_data/train.json
wget https://s3.amazonaws.com/my89public/quac/val.json -O QuAC_data/dev.json

# Download CoQA
mkdir -p CoQA_data
wget https://worksheets.codalab.org/rest/bundles/0xe3674fd34560425786f97541ec91aeb8/contents/blob/ -O CoQA_data/train.json
wget https://worksheets.codalab.org/rest/bundles/0xe254829ab81946198433c4da847fb485/contents/blob/ -O CoQA_data/dev.json


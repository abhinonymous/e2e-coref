#!/bin/bash

#$ -j y
#$ -l 'hostname=b1[12345678]*|c*' 
#$ -l mem_free=4G,ram_free=4G
#$ -cwd
#$ -M abhinavsingh282@gmail.com
#$ -e bbc_pb2_ep0_50_dev_extractcache.errlog
#$ -o bbc_pb2_ep0_50_dev_extractcache.log
#$ -N bbc_pb2_ep0_50_dev_extractcache
# -pe 4

conda activate py37 

test_data="/export/c01/pxia/je2eea/joint_edl/data/test.english.jsonlines"
dev_data="/export/c01/pxia/je2eea/joint_edl/data/dev.english.jsonlines"
train_data="/export/c01/pxia/je2eea/joint_edl/data/train.english.jsonlines"


#output
outdir="/export/c10/asingh/data/coref/bbc_pb2_ep0.50/"


#Model
bbc_pb2_ep0_25="/export/c10/asingh/models/bert/pb2_finetuned/7nearest-cased_1Epoch/pytorch_model_Ep0.25/"
bbc_pb2_ep0_50="/export/c10/asingh/models/bert/pb2_finetuned/7nearest-cased_1Epoch/pytorch_model_Ep0.50/"


tokenizer="bert-base-cased"
device_gpu='cpu'
model=$bbc_pb2_ep0_50
data=$dev_data

if [ "$device_gpu" = 'gpu' ]; then
	CUDA_VISIBLE_DEVICES=`free-gpu` python3 -u cache_bert.py -bert_model $model -tokenizer $tokenizer -data $data -outdir $outdir -device gpu 
else
	python3 -u cache_bert.py -bert_model $model -tokenizer $tokenizer -data $data -outdir $outdir -device cpu
fi



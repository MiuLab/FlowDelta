python3 run_quac.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file QuAC_data/train.json \
  --predict_file QuAC_data/dev.json \
  --train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --max_query_length 64 \
  --doc_stride 128 \
  --output_dir output_quac/ \
  --log_freq 1000 \
  

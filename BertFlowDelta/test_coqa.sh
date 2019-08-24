python3 run_coqa.py \
  --bert_model bert-base-uncased \
  --do_predict \
  --do_lower_case \
  --train_file CoQA_data/train.json \
  --predict_file CoQA_data/dev_span.json \
  --output_file pred.json \
  --train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_answer_length 35 \
  --output_dir output_coqa/ \


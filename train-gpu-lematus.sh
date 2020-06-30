export CUDA_VISIBLE_DEVICES=3
#onmt_train -data data-pp/en_gum-um -save_model demo-model -train_steps 5000 -report_every 100 -world_size 1 -gpu_ranks 0 &> train-gpu.log &


#dim_word=300 # encoder and decoder embedding size
#dim=100  #hidden_units
#optimizer="adadelta"
###########################################################
## calculate the nubel of different words
#n_words_src=($(wc -l ${modeldir}/data/train-sources.json))
#n_words_src=$((n_words_src-1))
#
#n_words_trg=($(wc -l ${modeldir}/data/train-targets.json))
#n_words_trg=$((n_words_trg-1))
###########################################################
#dropout=0.2
######################################################################
#patience=10 # early_stopping_n_epochs
#early_stopping_steps=($(wc -l ${datadir}/train-sources))
#early_stopping_steps=$((early_stopping_steps *${patience} / batch_size))
######################################################################

early_stopping_steps=10
batch_size=60

datadir = English-20-char-context

#####################################################################
#use the first 10 epochs as a burn-in period
burn_in_for_n_epochs=10
validBurnIn=($(wc -l ./data-nematus/${datadir}/train-sources))
validBurnIn=$((validBurnIn *${burn_in_for_n_epochs} / batch_size))
#####################################################################

onmt_train -data data-pp/en_gum-um\
  --save_model demo-model\
  --encoder_type brnn\
  --decoder_type rnn\
  --enc_layers 2\
  --dec_layers 2\
  --rnn_type GRU\
  --batch_size 60\
  --src_word_vec_size 300\
  --tgt_word_vec_size 300\
  --rnn_size 100\
  --optim "adadelta" \
  --dropout 0.2\
  --early_stopping ${early_stopping_steps}\
  --warmup_steps ${burn_in_period}\
  --train_steps 6000\
  --report_every 10 \
  --world_size 1 \
  --gpu_ranks 0

  #&> train-gpu.log &

















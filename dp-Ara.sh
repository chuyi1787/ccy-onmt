#!/bin/bash

datadir=Arabic-20-char-context

onmt_preprocess \
  -train_src data-nematus/${datadir}/train-sources\
  -train_tgt data-nematus/${datadir}/train-targets \
  -valid_src data-nematus/${datadir}/dev-sources \
  -valid_tgt data-nematus/${datadir}/dev-targets \
  -src_seq_length 75 \
  -tgt_seq_length 75 \
  -save_data data-pp/${datadir}
import os
import re
import sys

############## PROPS ##############

# Hiperparameters
a_layers = [3]
a_adam_beta1 = [0.9]
a_adam_beta2 = [0.998]
a_warmup_steps = [4000]
a_learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Train
log_dir = ".\\202207050915_log_test_learning_rate"

# Inference
beam_default_inference = 4
inference_source_train = "data/phoenix2014T.train.gloss"
inference_source_val = "data/phoenix2014T.dev.gloss"

# Scoring
bleu_default_scoring = 4
inference_goal_train = "data/phoenix2014T.train.de"
inference_goal_val = "data/phoenix2014T.dev.de"
###################################

# Make dirs if not exists
if not os.path.exists(log_dir):  
  os.makedirs(log_dir)


# Formats of the commands to train, inference and scoring
cmd_train  = ("python train.py -data data/dgs -save_model {model_file} -keep_checkpoint 1 "
  "-layers {layers} -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 "
  "-encoder_type transformer -decoder_type transformer -position_encoding "
  "-max_generator_batches 2 -dropout 0.1 "
  "-early_stopping 3 -early_stopping_criteria accuracy ppl "
  "-batch_size 2048 -accum_count 3 -batch_type tokens -normalization tokens "
  "-optim adam -adam_beta1 {adam_beta1} -adam_beta2 {adam_beta2} -decay_method noam -warmup_steps {warmup_steps} -learning_rate {learning_rate} "
  "-max_grad_norm 0 -param_init 0 -param_init_glorot "
  "-label_smoothing 0.1 -valid_steps 100 -save_checkpoint_steps 100 "
  "-world_size 1 -gpu_ranks 0 -log_file {log_file}"
)
log_file = log_dir + "\\log_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}.txt"
model_file = log_dir + "\\model_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}"

cmd_inference = "python translate.py -model {model} -src {src} -output {output} -gpu 0 -replace_unk -beam_size {beam_size}"
inference_output_train = log_dir + "\\inference_train_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"
inference_output_val = log_dir + "\\inference_val_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"

cmd_scoring_bleu = "python tools/bleu.py {bleu} {pred} {goal} > {log_file}"
scoring_bleu_train_log_file = log_dir + "\\score_train_bleu_{bleu}_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"
scoring_bleu_val_log_file = log_dir + "\\score_val_bleu_{bleu}_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"

cmd_scoring_rouge = "python tools/rouge.py {pred} {goal} > {log_file}"
scoring_rouge_train_log_file = log_dir + "\\score_train_rouge_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"
scoring_rouge_val_log_file = log_dir + "\\score_val_rouge_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}_{beam_size}.txt"

#### INIT TRAIN, INFERENCE, SCORING FUNCTIONS ####

def train(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate):
  log_file_formated = log_file.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate
  )
  model_file_formated = model_file.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate
  )
  cmd_train_formated = cmd_train.format(
    model_file=model_file_formated,
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    log_file=log_file_formated
  )
  os.system(cmd_train_formated)

def inference(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate):
  # Find the model generated
  dir_list = os.listdir(log_dir)
  model_file_name_to_search = "model_{layers}_{adam_beta1}_{adam_beta2}_{warmup_steps}_{learning_rate}".format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
  )
  final_model_file = log_dir + "\\" + next(filter(lambda f: model_file_name_to_search in f, dir_list))

  # Inference
  # Inference train
  inference_output_train_formated = inference_output_train.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_inference_train_formated = cmd_inference.format(
    model=final_model_file,
    src=inference_source_train,
    output=inference_output_train_formated,
    beam_size=beam_default_inference
  )
  os.system(cmd_inference_train_formated)
  # Inference val
  inference_output_val_formated = inference_output_val.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_inference_val_formated = cmd_inference.format(
    model=final_model_file,
    src=inference_source_val,
    output=inference_output_val_formated,
    beam_size=beam_default_inference
  )
  os.system(cmd_inference_val_formated)

def scoring(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate):
  # Scoring bleu
  # Scoring train
  inference_output_train_formated = inference_output_train.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  scoring_bleu_train_log_file_formated = scoring_bleu_train_log_file.format(
    bleu=bleu_default_scoring,
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_scoring_bleu_train_formated = cmd_scoring_bleu.format(
    bleu=bleu_default_scoring, 
    pred=inference_output_train_formated, 
    goal=inference_goal_train, 
    log_file=scoring_bleu_train_log_file_formated
  )
  os.system(cmd_scoring_bleu_train_formated)
  # Scoring val 
  inference_output_val_formated = inference_output_val.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  scoring_bleu_val_log_file_formated = scoring_bleu_val_log_file.format(
    bleu=bleu_default_scoring,
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_scoring_bleu_val_formated = cmd_scoring_bleu.format(
    bleu=bleu_default_scoring, 
    pred=inference_output_val_formated, 
    goal=inference_goal_val, 
    log_file=scoring_bleu_val_log_file_formated
  )
  os.system(cmd_scoring_bleu_val_formated)
  # Scoring rouge
  # scoring train
  scoring_rouge_train_log_file_formated = scoring_rouge_train_log_file.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_scoring_rouge_train_formated = cmd_scoring_rouge.format(
    pred=inference_output_train_formated, 
    goal=inference_goal_train, 
    log_file=scoring_rouge_train_log_file_formated
  )
  os.system(cmd_scoring_rouge_train_formated)
  # Scoring val 
  scoring_rouge_val_log_file_formated = scoring_rouge_val_log_file.format(
    layers=layers,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    beam_size=beam_default_inference
  )
  cmd_scoring_rouge_val_formated = cmd_scoring_rouge.format(
    pred=inference_output_val_formated, 
    goal=inference_goal_val, 
    log_file=scoring_rouge_val_log_file_formated
  )
  os.system(cmd_scoring_rouge_val_formated)

#### END TRAIN, INFERENCE, SCORING FUNCTIONS ####

# Main process for each hiperparameter
for layers in a_layers:
  for adam_beta1 in a_adam_beta1:
    for adam_beta2 in a_adam_beta2:
      for warmup_steps in a_warmup_steps:
        for learning_rate in a_learning_rate:      
          train(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate)
          inference(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate)
          scoring(layers, adam_beta1, adam_beta2, warmup_steps, learning_rate)
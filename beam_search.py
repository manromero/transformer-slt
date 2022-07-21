import os
import re
import sys

############## PROPS ##############
a_beam_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Model
checkpoint_path = '.\\202207042051_log_test_learning_rate\\model_3_0.9_0.998_4000_0.2_step_2800.pt'
# Data
inference_source_val = "data/phoenix2014T.dev.gloss"
inference_goal_val = "data/phoenix2014T.dev.de"
# Log
log_dir = '.\\beam_search'
###################################

cmd_inference = "python translate.py -model {model} -src {src} -output {output} -gpu 0 -replace_unk -beam_size {beam_size}"

cmd_scoring_bleu_4 = ("python tools/bleu.py 4 {pred} {goal} > {log_file}")
cmd_scoring_rouge = ("python tools/rouge.py {pred} {goal} > {log_file}")

def inference(generated_text_file, beam_size):
  # Inference val
  cmd_inference_val_formated = cmd_inference.format(
    model=checkpoint_path,
    src=inference_source_val,
    output=generated_text_file,
    beam_size=beam_size
  )
  os.system(cmd_inference_val_formated)

def scoring(generated_text_file, scoring_file_bleu4_txt, scoring_file_rouge_txt):
  cmd_scoring_bleu_4_formated = cmd_scoring_bleu_4.format(pred=generated_text_file, goal=inference_goal_val, log_file=scoring_file_bleu4_txt)
  os.system(cmd_scoring_bleu_4_formated)
  cmd_scoring_rouge_formated = cmd_scoring_rouge.format(pred=generated_text_file, goal=inference_goal_val, log_file=scoring_file_rouge_txt)
  os.system(cmd_scoring_rouge_formated)

# Make dirs if not exists
if not os.path.exists(log_dir):  
  os.makedirs(log_dir)

for beam_size in a_beam_size:
  generated_text_file = log_dir + '\\generated_text_valid_{beam_size}.txt'.format(beam_size=beam_size)
  # Inference
  inference(generated_text_file, beam_size)
  # Scoring 
  scoring_file_bleu4_txt = log_dir + "\\score_val_blue_{beam_size}.txt".format(beam_size=beam_size)
  scoring_file_rouge_txt = log_dir + "\\score_val_rouge_{beam_size}.txt".format(beam_size=beam_size)
  scoring(generated_text_file, scoring_file_bleu4_txt, scoring_file_rouge_txt)
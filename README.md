# One-word attack for text ranking

## Dependencies

  Will be updated later.

## Steps to reproduce

  ### 1. Models
  Save the Neural Ranker Models (NRMs) in the directory `./models/model_name/`.
  
  ### 2. Reranked passages
  Save the reranked passages in the directory `./reranked_files/`.
  The files should be in .tsv format `qid \t passid \t score \t rank`.
  Name of the file = `Model_name_dataset`, where dataset = `trec_dl_2019` or `trec_dl_2020` or `msmarco_dev_200` in our case.
  #### Original Passages
  Save the original passages in `./reranked_files/ori_pass_files/model_name_dataset`
  file format = `qid \t passid \t text`
  You can use `find_pass.py` to find the original texts for a ranked list.

  ### 3. Choose attack pass
  Run `choose_attack_pass.py`.
  #### Arguments
  * "--model_ranked_list_path" (path of the ranked list given by the model in the above-mentioned format)
  * "--starting_pos" (starting position of the ranked list from where the attacking of passages starts. default 11)
  * "--output_directory" (` ./Attack_pass/model_name_dataset`)

  ### 4. one_word_attack
  **one_word_grad / one_word_best_grad** \
  Run `st_bb_attack.py`.
  #### Imp Arguments
  * "--one_word" : (stores true) for one_word_attack.
  * "--embed_path" : default=`./models/counter-fitted-vectors.txt`.
  * "--boo" : (stores true) if used then the attack will be `one_word_best_grad` otherwise `one_word_grad`.
  * "--model_name" : name of the model (should be the same as the model name in the `models` directory. for example `--model_name Luyu`.
  * "--attack_dataset" : `trec_dl_2019` or `trec_dl_2020` or `msmarco_dev_200` in our case. for example `--attack_dataset trec_dl_2019`.
  * "--save_file_name" : give a name of the file where you want to save the attacked passages. for example `--save_file_name boo_01`.\


  Example Run
  
  ```bash
  python3 st_bb_attack.py --one_word --boo --model_name Luyu --attack_dataset trec_dl_2019 --save_file_name boo_01
  ```

  
  
  
  
    

# One Word Is Enough

Official implementation of the ECIR 2026 paper:

**"One Word Is Enough: Minimal Adversarial Perturbations for Neural Text Ranking"**

---

## 📄 Paper

- 📚 Springer (ECIR 2026): https://link.springer.com/chapter/10.1007/978-3-032-21300-6_31  
- 📄 arXiv (preprint): https://arxiv.org/abs/2601.20283  

---

## 🧠 Overview

Neural ranking models are widely used in modern IR systems.  
This work demonstrates a surprising and important vulnerability:

> **A single carefully chosen word can significantly alter ranking outcomes.**

We study **minimal adversarial perturbations**, showing that even extremely small changes—just *one word* can:

- Degrade ranking performance  
- Manipulate document ordering  
- Expose robustness issues in neural rankers  

---

## Key Contributions

- 🔹 Introduce **single-word adversarial perturbations** for ranking models
- 🔹 Compares better with adversarial attack baselines such as PRADA 
- 🔹 Show strong impact on models like BERT-based and monoT5 rankers
- 🔹 Provide systematic evaluation on TREC DL 2019, 2020, and MSMARCO dev set
- 🔹 Reveals a Goldilocks zone in mid-ranked documents
---

## 📊 Dataset & Benchmarks

Experiments are conducted on standard datasets such as:

- TREC Deep Learning (DL) 2019, 2020 passage ranking
- MS MARCO Dev  

---

## 📂 Repository Structure

```bash
one_word/
│── data/               # datasets
│── models/             # model implementations
│── scripts/            # training / evaluation scripts
│── attack.py           # adversarial attack logic
│── evaluate.py         # evaluation scripts
│── requirements.txt
```
---

## ⚙️ Installation

```bash
git clone https://github.com/Tanmay-Karmakar/one_word.git
cd one_word

# create environment (recommended)
conda create -n oneword python=3.9
conda activate oneword
```
---

## ▶️ Usage

### 1. Models
Save the Neural Ranker Models (NRMs) in the directory:
`./models/model_name/`


### 2. Reranked passages

Save the reranked passages in the directory `./reranked_files/`.
The files should be in .tsv format `qid \t passid \t score \t rank`.
Name of the file = `Model_name_dataset`, where dataset = `trec_dl_2019` or `trec_dl_2020` or `msmarco_dev_200` in our case.

#### Original Passages

Save the original passages in `./reranked_files/ori_pass_files/model_name_dataset`
file format = `qid \t passid \t text`
You can use `find_pass.py` to find the original texts for a ranked list.


### 3. Choose attack passages
Run: 
```bash
python choose_attack_pass.py
```
Arguments:
- `--model_ranked_list_path` : path of the ranked list given by the model in the above-mentioned format
- `--starting_pos` : starting position of the ranked list from where the attacking of passages starts (default 11)
- `--output_directory` : `./Attack_pass/model_name_dataset`

### one word attack
```bash
python st_bb_attack.py
```
Important Arguments:
- `--one_word` : (stores true) for one_word_attack.
- `--embed_path` : default=`./models/counter-fitted-vectors.txt`.
- `--boo` : (stores true) if used then the attack will be `one_word_best_grad` otherwise `one_word_grad`.
- `--model_name` : name of the model (should be the same as the model name in the `models` directory. for example `--model_name Luyu`.
- `--attack_dataset` : `trec_dl_2019` or `trec_dl_2020` or `msmarco_dev_200` in our case. for example `--attack_dataset trec_dl_2019`.
- `--save_file_name` : give a name of the file where you want to save the attacked passages. for example `--save_file_name boo_01`.
- `--no_cuda` : (stores true) if used, the program will run on CPU.
- `--previous_done` : (integer value) if some queries were done previously, give the number of queries.
  Rest of the arguments you may use as they are in default.

Example Run : 

```bash
  python3 st_bb_attack.py --one_word --boo --model_name Luyu --attack_dataset trec_dl_2019 --save_file_name boo_01 --previous_done 15
```

### one_word_start / one_word_sim
Run `start_st_bb_attack.py` / `sim_st_bb_attack.py` 

Important Arguments : 
- `--model_name` : name of the model (should be the same as the model name in the `models` directory. for example `--model_name Luyu`.
- `--attack_dataset` : `trec_dl_2019` or `trec_dl_2020` or `msmarco_dev_200` in our case. for example `--attack_dataset trec_dl_2019`.
- `--save_file_name` : give a name of the file where you want to save the attacked passages. for example `--save_file_name start_01`.
- `--orig_pass_file` : give the original passage file path. for example `--orig_pass_file ./reranked_files/ori_pass_files/trec_dl_2019`(optional)
- `--no_cuda` : (stores true) if used, the program will run on CPU.
- `--previous_done` : (integer value) if some queries were done previously, give the number of queries.\

Example Run: 
  
```bash
  python3 start_st_bb_attack.py --model_name Luyu --attack_dataset trec_dl_2019 --save_file_name boo_01 --previous_done 15
```
```bash
  python3 sim_st_bb_attack.py --model_name Luyu --attack_dataset trec_dl_2019 --save_file_name boo_01 --previous_done 15
```


---

## 📌 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-21300-6_31,
author="Karmakar, Tanmay
and Saha, Sourav
and Majumdar, Debapriyo
and Halder, Surjyanee",
title="One Word Is Enough: Minimal Adversarial Perturbations for Neural Text Ranking",
booktitle="Advances in Information Retrieval",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="409--417",
isbn="978-3-032-21300-6"
}
```

---


## 📬 Contact

For questions or collaborations, feel free to reach out.

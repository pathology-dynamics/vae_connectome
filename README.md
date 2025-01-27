# Reproduction of results


## Installation

```bash
conda create -n [env_name] python=3.12
pip install -r requirements.txt
cd src
```

## Run the Experiments on synthetic datasets

```bash
bash synthetic.sh
```
## Run the Experiments on Adni-AV45-PET

```bash
bash adni_3d.sh
```

## Run the Experiments Adni-1451-PET

```bash
bash adni_1451_run.sh
```


## An example of ablation study of soft constraints

```bash
python main.py --rid -1 --Nbiom 4 --random-seed -1 --sample-size 100 --epochs 900 \
	--lr 0.01 --verbal 1 --prior eye --beta 0.1 --checkpoint 100 --model-name no-source \
	--Pathological -1 --train-len 2 --validation-len 1 --experiments 10 --soft-constr 0 \
	--adaptive-laplacian 1 --vmax 0.2
```
python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name no-source \
	--Pathological 1 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name no-source \
	--Pathological 2 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2

python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name no-source \
	--Pathological 3 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2

python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name LinearSource \
	--Pathological 1 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2

	
python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name LinearSource \
	--Pathological 2 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name LinearSource \
	--Pathological 3 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name ExpSource \
	--Pathological 1 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2

	
python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name ExpSource \
	--Pathological 2 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python orchestrator_adni.py --data-path ../data/pseudo_adni_cleaned.csv  --rid -1 --Nbiom 4 --random-seed -1  \
	--sample-size 100 --epochs 900  --lr 0.01 --verbal 0 --prior eye --beta 1 --checkpoint 100 --model-name ExpSource \
	--Pathological 3 --train-len 2 --validation-len 0 --experiments 20 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2



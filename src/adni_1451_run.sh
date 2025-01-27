python orchestrator_adni1451.py --rid -1 --random-seed -1 --sample-size 100 --epochs 2100 \
	--lr 0.005 --verbal 1 --prior eye --beta 0.01 --checkpoint 100 --model-name no-source \
	--train-len 2 --soft-constr 0 --save-folder adni1451 --data-path ../data/UCBERKELEYAV1451_04_26_22_10Jul2024.csv \
	--name-path ../data/TauRegionList.csv --vmax 0.01 

python orchestrator_adni1451.py --rid -1 --random-seed -1 --sample-size 100 --epochs 2100 \
	--lr 0.005 --verbal 1 --prior eye --beta 0.01 --checkpoint 100 --model-name LinearSource \
	--train-len 2 --soft-constr 1 --save-folder adni1451 --data-path ../data/UCBERKELEYAV1451_04_26_22_10Jul2024.csv \
	--name-path ../data/TauRegionList.csv --vmax 0.01 


python orchestrator_adni1451.py --rid -1 --random-seed -1 --sample-size 100 --epochs 2100 \
	--lr 0.005 --verbal 1 --prior eye --beta 0.01 --checkpoint 100 --model-name ExpSource \
	--train-len 2 --soft-constr 1 --save-folder adni1451 --data-path ../data/UCBERKELEYAV1451_04_26_22_10Jul2024.csv \
	--name-path ../data/TauRegionList.csv --vmax 0.01 
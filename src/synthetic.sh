python main.py --rid -1 --Nbiom 4 --random-seed -1 --sample-size 100 --epochs 900 \
	--lr 0.01 --verbal 1 --prior eye --beta 0.1 --checkpoint 100 --model-name no-source \
	--Pathological -1 --train-len 2 --validation-len 1 --experiments 10 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python main.py --rid -1 --Nbiom 4 --random-seed -1 --sample-size 100 --epochs 900 \
	--lr 0.01 --verbal 1 --prior eye --beta 0.1 --checkpoint 100 --model-name no-source \
	--Pathological 0 --train-len 2 --validation-len 1 --experiments 10 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


python main.py --rid -1 --Nbiom 4 --random-seed -1 --sample-size 100 --epochs 900 \
	--lr 0.01 --verbal 1 --prior eye --beta 0.1 --checkpoint 100 --model-name LinearSource \
	--Pathological -1 --train-len 2 --validation-len 1 --experiments 10 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2

python main.py --rid -1 --Nbiom 4 --random-seed -1 --sample-size 100 --epochs 900 \
	--lr 0.01 --verbal 1 --prior eye --beta 0.1 --checkpoint 100 --model-name ExpSource \
	--Pathological -1 --train-len 2 --validation-len 1 --experiments 10 --soft-constr 1 \
	--adaptive-laplacian 1 --vmax 0.2


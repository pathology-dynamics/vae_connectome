# python reverse_cov.py --rid -1 --Nbiom 4 --random-seed 111 --vmax 1 --model-name LinearSource \
# 	--Pathological 1 --window-len 6

# python reverse_cov.py --rid -1 --Nbiom 4 --random-seed 111 --vmax 1 --model-name LinearSource \
# 	--Pathological 0 --window-len 6

python reverse_cov.py --rid -1 --Nbiom 4 --random-seed 111 --vmax 1 --model-name no-source \
	--Pathological 1 --window-len 6

python reverse_cov.py --rid -1 --Nbiom 4 --random-seed 111 --vmax 1 --model-name no-source \
	--Pathological 0 --window-len 6

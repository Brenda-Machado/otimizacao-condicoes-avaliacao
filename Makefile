venv/bin/activate: requirements.txt
	 python3 -m venv venv
	 ./venv/bin/pip install -r requirements.txt	

run: venv/bin/activate
	 ./venv/bin/python3 src/main.py

cp: venv/bin/activate
	 ./venv/bin/python3 src/new_otim_cp.py

pen:venv/bin/activate
	 ./venv/bin/python3 src/new_otim_pd.py

run_opt: venv/bin/activate
	 ./venv/bin/python3 src/adaptive_experiment_optimizer.py

teste: venv/bin/activate
	 ./venv/bin/python3 src/exp_cp_sdq.py

clean:
	 rm -rf __pycache__
	 rm -rf venv
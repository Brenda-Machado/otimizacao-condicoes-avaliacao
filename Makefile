venv/bin/activate: requirements.txt
	 python3 -m venv venv
	 ./venv/bin/pip install -r requirements.txt	

run: venv/bin/activate
	 ./venv/bin/python3 src/run_experiments_boxplot.py

opt_cp: venv/bin/activate
	 ./venv/bin/python3 src/cartpole_experiments.py

opt_pen:venv/bin/activate
	 ./venv/bin/python3 src/pendulum_experiments.py
	
opt_irace: venv/bin/activate
	 ./venv/bin/python3 src/run_irace_advanced.py

clean:
	 rm -rf __pycache__
	 rm -rf venv
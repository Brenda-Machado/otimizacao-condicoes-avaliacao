venv/bin/activate: requirements.txt
	 python3 -m venv venv
	 ./venv/bin/pip install -r requirements.txt	

run: venv/bin/activate
	 ./venv/bin/python3 src/main.py

cp: venv/bin/activate
	 ./venv/bin/python3 src/otim_cart_pole.py

pen:venv/bin/activate
	 ./venv/bin/python3 src/otim_pendulum.py

clean:
	 rm -rf __pycache__
	 rm -rf venv
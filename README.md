# fastapi-llmdet-test
## Setup
Create a new debian instance\

Update instance
```sh
apt update
apt upgrade
```

Setup python
```sh
apt install python3-pip -y
apt install python3.11-venv
```
Install git
```sh
apt install git
```

Create virtual env
```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install fastapi and LLMDet
```sh
pip install fastapi
pip install datasets
        
pip install git+https://github.com/Liadrinz/transformers-unilm
pip install git+https://github.com/huggingface/transformers


git clone https://github.com/cdellinger/fastapi-llmdet-test.git

cd fastapi-llmdet-test/LLMDet
python setup.py install
#pip install -r requirements.txt
cd ..
```

Start the server
```sh
fastapi run test.py
```

## Test
```sh
curl -d '{"raw_text":"abcdef"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8000
```
@echo off
setlocal
if not exist .venv\Scripts\python.exe python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python train.py
streamlit run app.py

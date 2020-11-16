pip3 install virtualenv
python3 -m venv ~/classify_env
source ~/classify_env/bin/activate
pip3 install tflite
pip3 install open-cv
pip3 install captcha
pip3 install scikit-build
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-linux_armv7l.whl
git clone https://github.com/snehakonoth/CaptchaDetection.git
cd CaptchaDetection
python3 classify.py --model-name test --captch-dir KONOTHS-project2rpi --output stuff.txt --symbols symbols.txt
git add stuff.txt
git push
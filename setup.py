from setuptools import setup
import py2exe

setup(
    name="MyApplication",
    version="1.0",
    description="My Python App",
    author="Your Name",
    console=['main.py'],  # Replace 'your_script.py' with your actual filename
    options={
        'py2exe': {
            'packages': ['google-auth', 'google-auth-oauthlib', 'google-auth-httplib2', 'google-api-python-client', 'tensorflow', 'tensorflow-hub', 'ultralytics', 'opencv-contrib-python', 'matplotlib', 'torch', 'torchvision', 'pytesseract', 'easyocr', 'reportlab', 'numpy', 'scipy'],
            'bundle_files': 1,  # Bundle dependencies into a single .exe
            'compressed': True,  # Compress the executable
        }
    },
    zipfile=None
)

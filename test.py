import subprocess
import os

subprocess.run("export GOOGLE_API_KEY='AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'", shell=True, executable="/bin/bash")
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBadUb2oZd7KjS8eY6XH8-AbMhO48nEs0g'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = 'python'
subprocess.run("export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'", shell=True, executable="/bin/bash")
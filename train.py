import os
from os import environ as export

commands = ['mkdir new_workspace', 'cd new_workspace',
            'clone --q https://github.com/tensorflow/models.git',
            'pip install -qq Cython contextlib2 pillow lxml matplotlib',
            'pip install -qq pycocotools',
            'wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip -c',
            'unzip protoc-3.13.0-linux-x86_64.zip -d protoc',
            'pwd = !pwd',
            'export[' + 'PATH]' + '+=' + ':' + 'pwd[0]' + '/protoc/bin',
            '(cd models/research;protoc object_detection/protos/*.proto --python_out=.)',
            'cd ..',
            'cd ..',
            'conda install --yes --prefix {sys.prefix} -c anaconda cudatoolkit=11.0 absl-py=0.10',
            'python3 -m pip install tensorflow-gpu==2.4 python-util==1.2.1 absl-py',
            'mkdir cuDNN8',
            'tar -xzvf cudnn-11.2-linux-x64-v8.1.0.77.tgz -C cuDNN8',
            'chmod a+x cuDNN8/cuda/include/cudnn*.h cuDNN8/cuda/lib64/libcudnn*',
            'cp models/research/object_detection/packages/tf2/setup.py models/research/',
            'python3 -m pip install .'
            'pip install opencv-python-headless==4.1.2.30',
            'export[' + 'LD_LIBRARY_PATH]' + '=cuDNN8/cuda/lib64/:',
            'python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"'
            ]

for command in commands:
    os.system(command)
    print(command)
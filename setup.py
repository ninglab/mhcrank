# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import re
import sys

from setuptools import setup

# normally we would import six.PY2 but can't yet assume that six
# is installed here
PY2 = (sys.version_info.major == 2)

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
except:
    logging.warning("Failed to load %s" % readme_filename)
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    logging.warning("Conversion of long_description from MD to RST failed")
    pass

with open('mhcrank/version.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

if __name__ == '__main__':
    required_packages = [
        'six',
        'pandas>=0.20.3',
        'appdirs',
        'tensorflow>=2.2.0,<2.3.0',
        'scikit-learn',
        'mhcnames',
        'pyyaml',
        'tqdm',
        'np_utils',
    ]

    setup(
        name='mhcrank',
        version=version,
        description="MHC Class I Presentation Predictor",
        author="Patrick J Lawrence and Xia Ning",
        author_email="patrick.skillman-lawrence@osumc.edu",
        url="https://github.com/ninglab/mhcrank",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=required_packages,
        long_description=readme,
        packages=[
            'mhcrank',
        ],
    )

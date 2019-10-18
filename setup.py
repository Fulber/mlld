from distutils.core import setup

setup(
    name='LinkDetection',
    version='0.1.0',
    author='N. Ovidiu',
    author_email='jrh@example.com',
    packages=['towelstuff', 'towelstuff.test'],
    scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    license='LICENSE.txt',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
		'certifi == 2018.1.18',
		'chardet == 3.0.4',
		'idna == 2.6',
		'numpy == 1.14.1',
		'requests == 2.20.0',
		'scikit-learn == 0.19.1',
		'scipy == 1.0.0',
		'urllib3 == 1.22'
	],
)
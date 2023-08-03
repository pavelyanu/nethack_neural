from setuptools import setup, find_packages

setup(
    name='nethack_neural',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'gym',
        'minihack',
        'nle',
        'numpy',
        'torch',
        'tqdm',
        'pandas',
        'matplotlib',
        'plotext'
    ],
    entry_points={
        'console_scripts': [
            'nethack_neural=nethack_neural.main:cli',
        ],
    }

)
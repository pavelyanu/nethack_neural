from setuptools import setup, find_packages

setup(
    name='nethack_neural',
    version='0.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click==8.1.6',
        'gym==0.23.0',
        'minihack==0.1.5',
        'nle==0.9.0',
        'numpy==1.25.2',
        'torch==2.0.1',
        'tqdm==4.65.0',
        'pandas==2.0.3',
        'matplotlib==3.7.2',
        'plotext==5.2.8'
    ],
    entry_points={
        'console_scripts': [
            'nethack_neural=nethack_neural.main:cli',
        ],
    }
)

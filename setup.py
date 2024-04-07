from setuptools import setup, find_packages

setup(
    name='grooveEvaluator',
    version='0.1.5',
    author='Daniel Flores',
    description="",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dafg05/grooveEvaluator',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'scikit-learn',
    ],
    classifiers=[
        # Choose classifiers as appropriate for your project
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.6',  # Specify compatible Python versions
)
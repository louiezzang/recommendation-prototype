from setuptools import setup, find_packages

setup(
    name='recommendation-engine',
    version='0.1',
    description='recommendation-engine',
    author='Younggue Bae',
    author_email='younggue.bae@mediacorp.com.sg',
    install_requires=[],
    packages=find_packages(exclude=[]),
    keywords=['recommendation'],
    python_requires='>=3',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

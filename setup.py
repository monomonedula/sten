from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name='pypardiso',
    version="0.1.0",
    packages=["simple_nodes_embedding"],
    author="Vladyslav Halchenko",
    author_email="valh@tuta.io",
    license="Apache License 2.0",
    url="https://github.com/monomonedula/simple-graph-embedding",
    long_description=open('README.md').read(),
    description='Simple deterministic algorithm for generating graph nodes topological embeddings.',
    classifiers=[
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords="graph embedding topology",
    # install_requires=["markdown"],
    # test_suite="nose.collector",
    # setup_requires=["pytest-runner"],
    # tests_require=["pytest"],
    long_description_content_type='text/markdown',
)

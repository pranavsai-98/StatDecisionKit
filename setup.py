from setuptools import setup, find_packages

setup(
    name='StatDecisionKit',
    version='0.1.0',
    author='Computer science foundations team 7',
    author_email='pranavsai98@gwu.edu',

    description="""In the rapidly evolving field of data science, the need for efficient and accurate data analysis is paramount. Our Python library, "StatDecisionKit," is designed to streamline and automate key aspects of the data science pipeline, focusing on statistical testing and feature selection.
    
    Our primary objective is to create a Python library that simplifies the process of statistical testing and feature selection for data scientists, researchers, and analysts. This tool is particularly beneficial for those without extensive statistical backgrounds.
    
    """,

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pranavsai-98/StatDecisionKit',
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

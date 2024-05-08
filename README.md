![Tests](https://github.com/2024-SWN90017-18/MT-Updating-Matilda/actions/workflows/validation-tests.yml/badge.svg)
[![Read the Docs](https://img.shields.io/readthedocs/matilda)](https://docs.matilda.neatht.com)


# 2. Updating the MATILDA [code:MT]
### (Masters Advanced Software Project SWEN90017)
## Client: Dr. Mario Andres Munoz Acosta

(Research fellow at ARC Centre in Optimisation Technologies, Integrated Methodologies and Applications (OPTIMA))

Project Mentor: Ben Golding

Updating the MATILDA (Melbourne Algorithm Test Instance Library with Data Analytics)

History is filled with dramatic examples of deployment disasters – exploding space rockets or deaths of patients from overdose by computerised radiation therapy  - caused by inadequate testing of algorithms, but sometimes the signs that an algorithm is not fit for purpose are more subtle. Ensuring that an algorithm has been tested under all possible realistic conditions is a significant challenge in both industrial and academic contexts. Since exhaustive testing of a potentially infinite set of conditions is usually not possible, a critical question arises: how can we rigorously and objectively evaluate the strengths and weaknesses of algorithms?

Standard practice for testing a new algorithm has long been criticized for its somewhat arbitrary and unscientific convention: collect some test instances to run on the algorithm; compare performance to some published algorithms in the literature; if the new algorithm’s performance is better on average across the chosen test instances, the paper is likely publishable; weaknesses of the algorithm are not required to be revealed if on-average performance is better. For more than 20 years, this criticism of the standard practice, and the need for a more experimental algorithmic science, has been widely acknowledged.

The Melbourne Algorithm Test Instance Library with Data Analytics (MATILDA)  (https://matilda.unimelb.edu.au/matilda/) is a cloud research platform for experimental algorithmics. It implements the Instance Space Analysis methodology, which supports the analysis and design of new algorithms by providing automated analysis of experimental data. It has been successfully applied across a variety of fields such as combinatorial optimisation, continuous black-box optimisation, machine learning, time series forecasting, and software testing. MATILDA provides, besides interactive tools, access to code and data from previous published studies. MATILDA has a JavaScript front-end and a MATILAB back-end, each one deployed on independent virtual machines, with one and sixteen cores respectively. Users can request accounts to access MATILDA, and submit jobs managed through SLURM.  It is being used for over 150 unique users worldwide, and over 10000 total views of its download page since 2019.

Unfortunately, MATILDA has limited maintenance and support. Hence, it is becoming unable to maintain and enlarge its user base. The objective of this project will be to significantly upgrade MATILDA's capabilities, both in its front and back ends. The main tasks will be:

To change the back-end codebase from MATLAB to Python.
To convert the job management system to independent VM, following industry best practice.
Implement better user and data management systems, including password recovery, two-factor authentication and other industry best practices.
Better integration with existing platforms, such as FigShare and GitHub.

# Development Environment Setup Guide

REQUIREMENTS: Python 3.12 installed

### Step 1: Install poetry

*Linux, Mac, WSL*

`curl -sSL https://install.python-poetry.org | python3 -`

*Windows*

`(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`

### Step 2: Setup virtual environment
`poetry shell`

### Step 3: Install python dependencies into virtual environment
`poetry install`

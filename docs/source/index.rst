.. RLPortfolio documentation master file, created by
   sphinx-quickstart on Fri Sep 20 02:03:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/CaioSBC/RLPortfolio

RLPortfolio Documentation
=========================

**RLPortfolio** is a Python package which provides several features to implement, train and test reinforcement learning agents that optimize a financial portfolio:

* A training simulation environment that implements the state-of-the-art mathematical formulation commonly used in the research field.
* Two policy gradient training algorithms that are specifically built to solve the portfolio optimization task.
* Four cutting-edge deep neural networks implemented in PyTorch that can be used as the agent policy.

.. note::

   This project is mainly intended for academic purposes. Therefore, be careful if using RLPortfolio to trade real money and consult a professional before investing, if possible.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/about
   getting_started/installation
   getting_started/components
   getting_started/first_agent

.. toctree::
   :maxdepth: 2
   :caption: Environment

   environment/environment
   environment/portfolio_optimization_env

.. toctree::
   :maxdepth: 2
   :caption: Policy

   policy/policy
   policy/implemented_policies

.. toctree::
   :maxdepth: 5
   :caption: Package Reference

   apidoc/modules

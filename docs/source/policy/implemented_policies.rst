Implemented Policies
====================

On this page, the policies already implemented in RLPortfolio are listed with a brief description and a link to its reference paper.

.. py:currentmodule:: rlportfolio.policy

.. list-table::
    :width: 100 %
    :header-rows: 1

    * - Policy
      - Description
      - Reference
    * - :py:class:`~eiie.EIIE`
      - Ensemble of Identical Independent Evaluators with convolutional neural networks.
      - `Link <https://doi.org/10.48550/arXiv.1706.10059>`__
    * - :py:class:`~eiie.EIIERecurrent`
      - Ensemble of Identical Independent Evaluators with recurrent neural networks.
      - `Link <https://doi.org/10.48550/arXiv.1706.10059>`__
    * - :py:class:`~ei3.EI3`
      - Ensemble of Identical Independent Inception. It adds multi-scale temporal feature extraction convolutional neural networks to EIIE.
      - `Link <https://doi.org/10.1145/3357384.3357961>`__
    * - :py:class:`~gpm.GPM`
      - Graph Convolution for Portfolio Management. Implements relational graph convolutional networks into EI3.
      - `Link <https://doi.org/10.1016/j.neucom.2022.04.105>`__
    * - :py:class:`~gpm.GPMSimplified`
      - Simplified version of GPM without the multi-scale temporal feature extraction with convolutional neural networks.
      - `Link <https://doi.org/10.1016/j.neucom.2022.04.105>`__

========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        | |codecov|
        | |codacy|
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/q100opt/badge/?style=flat
    :target: https://readthedocs.org/projects/q100opt
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/quarree100/q100opt.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/quarree100/q100opt

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/quarree100/q100opt?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/quarree100/q100opt

.. |codecov| image:: https://codecov.io/gh/quarree100/q100opt/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/quarree100/q100opt

.. |codacy| image:: https://img.shields.io/codacy/grade/[Get ID from https://app.codacy.com/app/quarree100/q100opt/settings].svg
    :target: https://www.codacy.com/app/quarree100/q100opt
    :alt: Codacy Code Quality Status

.. |commits-since| image:: https://img.shields.io/github/commits-since/quarree100/q100opt/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/quarree100/q100opt/compare/v0.0.0...master



.. end-badges

Model builder for oemof-solph optimisation models with a focus an district energy systems.

* Free software: MIT license

Installation
============

::

    pip install q100opt

You can also install the in-development version with::

    pip install https://github.com/quarree100/q100opt/archive/master.zip


Documentation
=============


https://q100opt.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox


.. image:: https://api.codacy.com/project/badge/Grade/6172cb1979214be2837a34df246668e4
   :alt: Codacy Badge
   :target: https://app.codacy.com/gh/quarree100/q100opt?utm_source=github.com&utm_medium=referral&utm_content=quarree100/q100opt&utm_campaign=Badge_Grade_Dashboard
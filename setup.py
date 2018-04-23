from setuptools import setup

setup(name='sklearn_helpers',
      version='0.0.2',
      description='Some Helpers for sklearn.',
      url='https://github.com/Root-App/sklearn-helpers',
      author='David E. Weirich',
      author_email='david.weirich@joinroot.com',
      license='MIT',
      packages=['sklearn_helpers'],
      install_requires=[
          'scikit-learn',
          'pandas',
          'numpy',
          'scipy'
      ],
      zip_safe=False)

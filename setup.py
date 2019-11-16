import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='mlrequest',
     version='1.1.3',
     author="Mathieu Rodrigue",
     author_email="support@mlrequest.com",
     description="Python client for the mlrequest machine learning API.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/mlrequest/mlrequest-python",
     packages=setuptools.find_packages(),
     install_requires=[
          'requests-futures',
          'requests',
          'sklearn-json'
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.5',
 )

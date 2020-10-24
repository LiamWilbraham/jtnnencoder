import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='jtnnencoder',
    version='0.1',
    license='MIT',
    description='An Easy to install version of the jtnn encoder for generation of latent molecule features.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Liam Wilbraham',
    author_email='liam.wilbrahaml@glasgow.ac.uk',
    url='https://github.com/LiamWilbraham/jtnnencoder',
    download_url='https://github.com/LiamWilbraham/jtnnencoder/archive/v_01.tar.gz',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy'
    ],
    keywords=['cheminformatics', 'chemistry'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)

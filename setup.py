import setuptools

with open("README.md") as f:
    long_desc = f.read()

setuptools.setup(name="habitat_sim2real_wgw101",
                 version="0.0.1",
                 author="Guillaume Bono",
                 author_email="guillaume.bono@insa-lyon.fr",
                 description="Extension of habitat to use with a robot running ROS",
                 long_description=long_desc,
                 long_description_content_type="text/markdown",
                 url="https://github.com/wgw101/habitat_sim2real",
                 packages=setuptools.find_packages("./src"),
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"],
                 python_requires=">=3.6")

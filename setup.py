from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return the list of requirements
    ''' 
    requirements= []
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name= "MLProject",
    version= "0.0.1",
    author= "Nikhil",
    author_email= "hellnix.47@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)
from glob import glob
from setuptools import setup
setup(
    name="coramo_bringup", version="0.1.0", packages=[],
    data_files=[("share/ament_index/resource_index/packages", ["resource/coramo_bringup"]),
                ("share/coramo_bringup", ["package.xml"]),
                ("share/coramo_bringup/launch", glob("launch/*.py")),
                ("share/coramo_bringup/params", glob("params/*.yaml"))],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Felipe Ballesteros", maintainer_email="felipe1024@gmail.com",
    description="Lanzadores y perfiles de CORAMO.", license="MIT",
)

from setuptools import find_packages, setup
setup(
    name="coramo_brain", version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[("share/ament_index/resource_index/packages", ["resource/coramo_brain"]),
                ("share/coramo_brain", ["package.xml"])],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Felipe Ballesteros", maintainer_email="felipe1024@gmail.com",
    description="Cerebro de CORAMO: voz a accion.", license="MIT",
    entry_points={"console_scripts": ["safety = coramo_brain.nodes.safety_node:main", "body_bridge_sim = coramo_brain.nodes.body_bridge_sim_node:main", "tts = coramo_brain.nodes.tts_node:main", "speech = coramo_brain.nodes.speech_node:main"]},
)

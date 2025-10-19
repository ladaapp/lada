from setuptools import setup, find_packages
from lada import VERSION

setup(
    name="lada",
    version=str(VERSION),
    description="Remove and recover pixelated areas in adult videos",
    python_requires='>=3.12',
    packages=find_packages(where='.',include=['lada','lada.*']),
    # ultralytics: pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
    # av: Binary wheels before 15.0.0 had either no nvidia encoders or libx265 was broken/missing.
    install_requires=['torch', 'ultralytics==8.3.203', 'numpy', 'opencv-python', 'tqdm', 'av>=15.0.0'],
    extras_require={
        'deepmosaics': ['scikit-image'],
        'basicvsrpp': ['mmengine==0.10.7', 'torchvision'], # mmengine pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
        'gui': ['pycairo', 'PyGObject'],
        'gui-dev': ['pygobject-stubs'],
        'training': ['torchvision', 'albumentations', 'tensorboard', 'standard-imghdr'],
        'dataset-creation': ['lap>=0.5.12', 'timm', 'einops', 'torchvision', 'pillow']
    },
    include_package_data=True,
    package_data={
        '': ['*.css', '*.ui', '*.gresource', '*.gresource.xml', '*.mo']
    },
    entry_points={
        'console_scripts': [
            'lada = lada.gui.main:main',
            'lada-cli = lada.cli.main:main'
        ],
    }
)

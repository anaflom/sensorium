import glob
import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class F2pyExtension(Extension):
    """Placeholder extension for Fortran sources compiled with f2py."""

    def __init__(self, name, sourcefile):
        super().__init__(name, sources=[])
        self.sourcefile = sourcefile


class BuildF2py(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, F2pyExtension):
            super().build_extension(ext)
            return

        module_name = ext.name.split(".")[-1]
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path) or ".", exist_ok=True)

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        sourcefile = os.path.abspath(ext.sourcefile)
        subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-c", "-m", module_name, sourcefile],
            check=True,
            cwd=build_temp,
        )

        so_files = glob.glob(os.path.join(build_temp, f"{module_name}*.so"))
        so_files += glob.glob(os.path.join(build_temp, f"{module_name}*.pyd"))
        if not so_files:
            raise RuntimeError(
                f"f2py compilation produced no output for {module_name}"
            )
        shutil.copy(so_files[0], ext_path)


setup(
    ext_modules=[
        F2pyExtension("ssdatam.assign_grid", "ssdatam/assign_grid.f90"),
    ],
    cmdclass={"build_ext": BuildF2py},
)

from distutils.core import setup, Extension

# name of module
name = "seobnre"

# version of module
version = "0.1"

# specify the name of the extension and source files
# required to compile this
ext_modules = Extension(
      name='_seobnre',
      sources=[
            "seobnre.i",
            "seobnre.c",
      ],
)

setup(name=name,
      version=version,
      ext_modules=[ext_modules])
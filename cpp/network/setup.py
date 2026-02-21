from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="embedding_cpp",
    ext_modules=[
        CppExtension(
            name="embedding_cpp",
            sources=[
                "src/PositionalEmbedding.cpp",
                "src/positional_embedding_bindings.cpp",
                "src/settings.cpp",
            ],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

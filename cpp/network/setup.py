from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sysconfig

python_include = sysconfig.get_paths()["include"]
setup(
    name="embedding_cpp",
    ext_modules=[
        CppExtension(
            name="embedding_cpp",
            sources=[
                "src/bindings.cpp",
                "src/PositionalEmbedding.cpp",
                "src/MultiHeadAttention.cpp",
            ],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

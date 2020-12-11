import os
import pkgutil


def compile_modules():
    from numba.pycc import CC

    cc = CC("aot")

    # collect AOT-compiled modules
    directory = os.path.dirname(os.path.abspath(__file__))
    for importer, module_name, _ in pkgutil.walk_packages([directory]):
        if module_name not in {"cc", "aot"}:
            module = importer.find_module(module_name).load_module(module_name)
            for function_name, (function, signature) in module.exports.items():
                cc.export(function_name, signature)(function)

    cc.compile()


def is_out_of_date(module):
    # Check if module is older than files in pymatting_aot
    module_modification_time = os.path.getmtime(module.__file__)

    directory = os.path.dirname(os.path.abspath(__file__))
    for importer, module_name, _ in pkgutil.walk_packages([directory]):
        loader = importer.find_module(module_name)
        if os.path.getmtime(loader.path) > module_modification_time:
            return True

    return False


# Test if modules need to be compiled
try:
    import pymatting_aot.aot

    # Test if modules need to be recompiled
    if is_out_of_date(pymatting_aot.aot):
        print("Recompiling modules because they are out of date.")
        print("This might take a minute.")
        compile_modules()

        import importlib

        importlib.reload(pymatting_aot.aot)

except ImportError:
    print("Failed to import ahead-of-time-compiled modules.")
    print("This is expected on first import.")
    print("Compiling modules and trying again.")
    print("This might take a minute.")

    compile_modules()

    import pymatting_aot.aot

    print("Successfully imported ahead-of-time-compiled modules.")

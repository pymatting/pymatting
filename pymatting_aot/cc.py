def compile_modules():
    import os
    import pkgutil
    from numba.pycc import CC

    cc = CC("aot")

    # collect AOT-compiled modules
    directory = os.path.dirname(os.path.abspath(__file__))
    for importer, module_name, _ in pkgutil.walk_packages([directory]):
        if module_name != "cc":
            module = importer.find_module(module_name).load_module(module_name)
            for function_name, (function, signature) in module.exports.items():
                cc.export(function_name, signature)(function)

    cc.compile()


# Test if modules need to be compiled
try:
    import pymatting_aot.aot
except ImportError:
    print(
        "Failed to import ahead-of-time-compiled modules. This is expected on first import."
    )
    print("Compiling modules and trying again (this might take a minute).")

    compile_modules()

    import pymatting_aot.aot

    print("Successfully imported ahead-of-time-compiled modules.")

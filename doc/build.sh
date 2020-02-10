SPHINX_APIDOC_OPTIONS=members,no-undoc-members,show-inheritance sphinx-apidoc -f -o source/ ../pymatting/

rm source/modules.rst

python3 remove_captions.py source/pymatting.alpha.rst
python3 remove_captions.py source/pymatting.config.rst
python3 remove_captions.py source/pymatting.cutout.rst
python3 remove_captions.py source/pymatting.foreground.rst
python3 remove_captions.py source/pymatting.laplacian.rst
python3 remove_captions.py source/pymatting.preconditioner.rst
python3 remove_captions.py source/pymatting.rst
python3 remove_captions.py source/pymatting.solver.rst
python3 remove_captions.py source/pymatting.util.rst


sed 's/pymatting package/API Reference/' source/pymatting.rst > temp && mv temp source/pymatting.rst

make clean
make html


all: data small big

data:
	python experiment/data.py

small:
	python experiment/train_core.py -e 10 -i 0 -ds author

big: 
	python experiment/train_augmented.py -e 10 -i 0 -nds content -im author

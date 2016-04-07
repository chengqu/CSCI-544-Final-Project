
all: small big

small:
	python experiment/train_core.py -e 2

big: 
	python experiment/train_augmented.py

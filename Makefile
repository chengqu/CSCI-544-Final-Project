
all: small big

small:
	python experiment/train_core.py

big: 
	python experiment/train_augmented.py

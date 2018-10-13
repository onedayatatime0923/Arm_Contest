binaryClassifier.d: ../../include/binaryClassifier.h 
../../include/binaryClassifier.h: binaryClassifier.h
	@rm -f ../../include/binaryClassifier.h
	@ln -fs ../src/binaryClassifier/binaryClassifier.h ../../include/binaryClassifier.h

moveClassifier.d: ../../include/moveClassifier.h 
../../include/moveClassifier.h: moveClassifier.h
	@rm -f ../../include/moveClassifier.h
	@ln -fs ../src/moveClassifier/moveClassifier.h ../../include/moveClassifier.h

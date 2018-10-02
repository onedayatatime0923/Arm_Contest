classifier.d: ../../include/classifier.h 
../../include/classifier.h: classifier.h
	@rm -f ../../include/classifier.h
	@ln -fs ../src/classifier/classifier.h ../../include/classifier.h

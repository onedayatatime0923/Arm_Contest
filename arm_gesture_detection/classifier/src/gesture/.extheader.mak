gesture.d: ../../include/gesture.h 
../../include/gesture.h: gesture.h
	@rm -f ../../include/gesture.h
	@ln -fs ../src/gesture/gesture.h ../../include/gesture.h

dtw.d: ../../include/dtw.h 
../../include/dtw.h: dtw.h
	@rm -f ../../include/dtw.h
	@ln -fs ../src/dtw/dtw.h ../../include/dtw.h

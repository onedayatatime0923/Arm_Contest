util.d: ../../include/dtw.h ../../include/gesture.h ../../include/point.h 
../../include/dtw.h: dtw.h
	@rm -f ../../include/dtw.h
	@ln -fs ../src/util/dtw.h ../../include/dtw.h
../../include/gesture.h: gesture.h
	@rm -f ../../include/gesture.h
	@ln -fs ../src/util/gesture.h ../../include/gesture.h
../../include/point.h: point.h
	@rm -f ../../include/point.h
	@ln -fs ../src/util/point.h ../../include/point.h

LIBPKGS  = binaryClassifier moveClassifier util
MAIN     = main

LIBS     = $(addprefix -l, $(LIBPKGS))
LIBFILES = $(addsuffix .a, $(addprefix lib, $(LIBPKGS)))

dummy:
	@echo "Error: please use make <dtw | b | m | data | clean | cleanall>"

.PHONY : dtw b m data

dtw: CMD  = dtw
b: CMD  = binaryClassifier
m: CMD  = moveClassifier
data: CMD  = data

dtw: CMDFLAG = -DTEST_DTW
b: CMDFLAG = -DTEST_BINARYCLASSIFIER
m: CMDFLAG = -DTEST_MOVECLASSIFIER
data: CMDFLAG = -DTEST_DATA

dtw b m data: all
	
EXEC     = Test.$(CMD)

all: libs main

libs:
	@for pkg in $(LIBPKGS); \
	do \
		echo "Checking $$pkg..."; \
		cd src/$$pkg; \
                make -f make.$$pkg --no-print-directory PKGNAME=$$pkg; \
		cd ../..; \
	done

main:
	@echo "Checking $(MAIN)..."
	@cd src/$(MAIN); \
		make -f make.$(MAIN) --no-print-directory CMDFLAG=$(CMDFLAG) \
		                          INCLIB="$(LIBS)" EXEC=$(EXEC);
	@ln -fs bin/$(EXEC) .
#	@strip bin/$(EXEC)

clean:
	@for pkg in $(LIBPKGS); \
	do \
		echo "Cleaning $$pkg..."; \
		cd src/$$pkg; \
                make -f make.$$pkg --no-print-directory PKGNAME=$$pkg clean; \
                cd ../..; \
	done
	@echo "Cleaning $(MAIN)..."
	@cd src/$(MAIN); make -f make.$(MAIN) --no-print-directory clean
	@echo "Removing $(LIBFILES)..."
	@cd lib; rm -f $(LIBFILES)
	@echo "Removing $(EXEC)..."
	@rm -f bin/$(EXEC)

cleanall:
	@for pkg in $(LIBPKGS); \
	do \
		echo "Cleaning $$pkg..."; \
		cd src/$$pkg; \
                make -f make.$$pkg --no-print-directory PKGNAME=$$pkg cleanall; \
                cd ../..; \
	done
	@echo "Cleaning $(MAIN)..."
	@cd src/$(MAIN); make -f make.$(MAIN) --no-print-directory cleanall
	@echo "Removing $(LIBFILES)..."
	@cd lib; rm -f $(LIBFILES)
	@echo "Removing $(EXEC)..."
	@rm -f bin/$(EXEC)
	@echo "Removing bin/*..."
	@rm -f bin/*
	@echo "binary file" > bin/READ.md

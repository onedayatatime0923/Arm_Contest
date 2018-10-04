LIBPKGS  = binaryClassifier moveClassifier util
MAIN     = main

LIBS     = $(addprefix -l, $(LIBPKGS))
LIBFILES = $(addsuffix .a, $(addprefix lib, $(LIBPKGS)))

dummy:
	@echo "Error: please use make <d | m | clean | cleanall>"

.PHONY : d m

d: CMD  = DTW
m: CMD  = merge

d: CMDFLAG = -DTEST_DTW
m: CMDFLAG = -DSORT_MERGE

d m: all
	
EXEC     = Classifier

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
